from __future__ import annotations
# gabion:decision_protocol_module
# gabion:boundary_normalization_module

import atexit
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Callable, Generator, List, Literal, Mapping, MutableMapping, Optional, TypeAlias, cast
import argparse
import io
import json
import os
import re
import subprocess
import sys

from click.core import ParameterSource
import typer
from gabion.cli_support.check.check_commands import (
    register_check_delta_bundle_command as _register_check_delta_bundle_command, register_check_group_callback as _register_check_group_callback, register_check_run_command as _register_check_run_command)
from gabion.cli_support.check.check_command_runtime import (
    check_raw_profile_args as _check_raw_profile_args_impl, run_check_aux_operation as _run_check_aux_operation_impl, run_check_command as _run_check_command_impl, run_check_raw_profile as _run_check_raw_profile_impl)
from gabion.cli_support.check.check_execution_plan import (
    check_derived_artifacts as _check_derived_artifacts_impl, build_check_execution_plan_request as _build_check_execution_plan_request_impl)
from gabion.cli_support.check.check_runtime import run_check as _run_check_impl
from gabion.cli_support.shared.dispatch_runtime import (
    dispatch_command as _dispatch_command_impl)
from gabion.cli_support.shared import (
    github_artifact_restore as _github_artifact_restore)
from gabion.cli_support.shared.parser_builder import dataflow_cli_parser as _build_dataflow_cli_parser
from gabion.cli_support.shared.raw_argparse import (
    parse_dataflow_args_or_exit as _parse_dataflow_args_or_exit_impl)
from gabion.cli_support.shared.payload_builder import (
    build_dataflow_payload as _build_dataflow_payload_impl)
from gabion.cli_support.shared.output_emitters import (
    emit_dataflow_result_outputs as _emit_dataflow_result_outputs_impl, write_lint_sarif as _write_lint_sarif_impl)
from gabion.cli_support.synth.synth_runtime import run_synth as _run_synth_impl
from gabion.cli_support.synth.synth_commands import (
    register_synth_command as _register_synth_command)
from gabion.cli_support.refactor.refactor_payload import (
    build_refactor_payload as _build_refactor_payload_impl)
from gabion.cli_support.refactor.refactor_runtime import (
    run_refactor_protocol as _run_refactor_protocol_impl)
from gabion.cli_support.shared.timeout_progress import (
    render_timeout_progress_markdown as _render_timeout_progress_markdown_impl)
from gabion.cli_support.shared.runtime_flags import (
    register_runtime_flags_callback as _register_runtime_flags_callback)
from gabion.cli_support.tooling_commands import (
    build_status_watch_options as _build_status_watch_options_impl,
    invoke_argparse_command as _invoke_argparse_command_impl,
    register_ci_watch_command as _register_ci_watch_command,
    register_tooling_passthrough_commands as _register_tooling_passthrough_commands,
    run_tooling_no_arg as _run_tooling_no_arg_impl,
    run_tooling_with_argv as _run_tooling_with_argv_impl,
    tooling_runner_override as _tooling_runner_override_impl,
)
from gabion.analysis.foundation.timeout_context import (
    check_deadline, deadline_loop_iter, render_deadline_profile_markdown)
from gabion.commands import (
    boundary_order, check_contract, command_ids, progress_contract as progress_timeline, transport_policy)
from gabion.runtime import deadline_policy, env_policy, path_policy, policy_runtime

DATAFLOW_COMMAND = command_ids.DATAFLOW_COMMAND
CHECK_COMMAND = command_ids.CHECK_COMMAND
SYNTHESIS_COMMAND = command_ids.SYNTHESIS_COMMAND
REFACTOR_COMMAND = command_ids.REFACTOR_COMMAND
STRUCTURE_DIFF_COMMAND = command_ids.STRUCTURE_DIFF_COMMAND
STRUCTURE_REUSE_COMMAND = command_ids.STRUCTURE_REUSE_COMMAND
DECISION_DIFF_COMMAND = command_ids.DECISION_DIFF_COMMAND
IMPACT_COMMAND = command_ids.IMPACT_COMMAND
LSP_PARITY_GATE_COMMAND = command_ids.LSP_PARITY_GATE_COMMAND
from gabion.lsp_client import (
    CommandRequest, run_command, run_command_direct)
from gabion.tooling.runtime import (
    ci_watch as tooling_ci_watch, tool_specs, run_dataflow_stage as tooling_run_dataflow_stage)
from gabion.tooling.delta import (
    delta_advisory as tooling_delta_advisory)
from gabion.tooling.docflow import (
    docflow_delta_emit as tooling_docflow_delta_emit)
from gabion.tooling.governance import (
    governance_audit as tooling_governance_audit, ambiguity_contract_policy_check as tooling_ambiguity_contract_policy_check, normative_symdiff as tooling_normative_symdiff)
from gabion.server_core import command_orchestrator_primitives
from gabion.tooling.impact import (
    impact_select_tests as tooling_impact_select_tests)
from gabion.json_types import JSONObject
from gabion.invariants import never
from gabion.order_contract import sort_once
from gabion.schema import (
    DecisionDiffResponseDTO, RefactorProtocolResponseDTO, LspParityGateResponseDTO, StructureDiffResponseDTO, StructureReuseResponseDTO, SynthesisPlanResponseDTO)
app = typer.Typer(add_completion=False)
check_app = typer.Typer(
    add_completion=False,
    help="Check command family.",
    invoke_without_command=True,
)
check_obsolescence_app = typer.Typer(
    add_completion=False,
    help="Test-obsolescence modalities.",
    invoke_without_command=True,
)
check_annotation_drift_app = typer.Typer(
    add_completion=False,
    help="Test-annotation-drift modalities.",
    invoke_without_command=True,
)
check_ambiguity_app = typer.Typer(
    add_completion=False,
    help="Ambiguity modalities.",
    invoke_without_command=True,
)
check_taint_app = typer.Typer(
    add_completion=False,
    help="Taint modalities.",
    invoke_without_command=True,
)
app.add_typer(check_app, name="check")
check_app.add_typer(check_obsolescence_app, name="obsolescence")
check_app.add_typer(check_annotation_drift_app, name="annotation-drift")
check_app.add_typer(check_ambiguity_app, name="ambiguity")
check_app.add_typer(check_taint_app, name="taint")
Runner: TypeAlias = Callable[..., JSONObject]
DEFAULT_RUNNER: Runner = run_command

CliRunDataflowRawArgvFn: TypeAlias = Callable[[list[str]], None]
CliRunCheckFn: TypeAlias = Callable[..., JSONObject]
CliRunSppfSyncFn: TypeAlias = Callable[..., int]
CliRunCheckDeltaGatesFn: TypeAlias = Callable[[], int]
CliRunCiWatchFn: TypeAlias = Callable[
    [tooling_ci_watch.StatusWatchOptions],
    tooling_ci_watch.StatusWatchResult,
]

_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")

_DEFAULT_TIMEOUT_TICKS = 100
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_CHECK_REPORT_REL_PATH = path_policy.DEFAULT_CHECK_REPORT_REL_PATH
_DEFAULT_STATUS_WATCH_ARTIFACT_ROOT = Path("artifacts/out/ci_watch")
_SPPF_GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
_SPPF_KEYWORD_REF_RE = re.compile(
    r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE
)
_SPPF_PLACEHOLDER_ISSUE_BY_COMMIT: dict[str, str] = {
    "683da24bd121524dc48c218d9771dfbdf181d6f0": "214",
    "61c5d617e7b1d4e734a476adf69bc92c19f35e0f": "214",
}

_LSP_PROGRESS_NOTIFICATION_METHOD = progress_timeline.LSP_PROGRESS_NOTIFICATION_METHOD
_LSP_PROGRESS_TOKEN_V2 = progress_timeline.LSP_PROGRESS_TOKEN_V2
_LSP_PROGRESS_TOKEN = _LSP_PROGRESS_TOKEN_V2
_STDOUT_ALIAS = "-"
_STDOUT_PATH = "/dev/stdout"


class CliTransportMode(str, Enum):
    lsp = "lsp"
    direct = "direct"


class CheckBaselineMode(str, Enum):
    off = "off"
    enforce = "enforce"
    write = "write"


class CheckGateMode(str, Enum):
    all = "all"
    none = "none"
    violations = "violations"
    type_ambiguities = "type-ambiguities"


class CheckLintMode(str, Enum):
    none = "none"
    line = "line"
    jsonl = "jsonl"
    sarif = "sarif"
    line_jsonl = "line+jsonl"
    line_sarif = "line+sarif"
    jsonl_sarif = "jsonl+sarif"
    all = "all"


class CheckStrictnessMode(str, Enum):
    high = "high"
    low = "low"


@dataclass(frozen=True)
class SppfSyncCommitInfo:
    sha: str
    subject: str
    body: str


@dataclass(frozen=True)
class CliDeps:
    run_dataflow_raw_argv_fn: CliRunDataflowRawArgvFn
    run_check_fn: CliRunCheckFn
    run_sppf_sync_fn: CliRunSppfSyncFn
    run_check_delta_gates_fn: CliRunCheckDeltaGatesFn
    run_ci_watch_fn: CliRunCiWatchFn


def _context_callable_dep(
    *,
    ctx: typer.Context,
    key: str,
    default: Callable[..., object],
) -> Callable[..., object]:
    obj = ctx.obj
    if not isinstance(obj, Mapping):
        return default
    candidate = obj.get(key)
    if candidate is None:
        return default
    if callable(candidate):
        return candidate
    never(
        "invalid cli dependency override",
        dependency=key,
        value_type=type(candidate).__name__,
    )
    return default  # pragma: no cover - never() raises


def _context_cli_deps(ctx: typer.Context) -> CliDeps:
    return CliDeps(
        run_dataflow_raw_argv_fn=cast(
            CliRunDataflowRawArgvFn,
            _context_callable_dep(
                ctx=ctx,
                key="run_dataflow_raw_argv",
                default=_run_dataflow_raw_argv,
            ),
        ),
        run_check_fn=cast(
            CliRunCheckFn,
            _context_callable_dep(
                ctx=ctx,
                key="run_check",
                default=run_check,
            ),
        ),
        run_sppf_sync_fn=cast(
            CliRunSppfSyncFn,
            _context_callable_dep(
                ctx=ctx,
                key="run_sppf_sync",
                default=_run_sppf_sync,
            ),
        ),
        run_check_delta_gates_fn=cast(
            CliRunCheckDeltaGatesFn,
            _context_callable_dep(
                ctx=ctx,
                key="run_check_delta_gates",
                default=_run_check_delta_gates,
            ),
        ),
        run_ci_watch_fn=cast(
            CliRunCiWatchFn,
            _context_callable_dep(
                ctx=ctx,
                key="run_ci_watch",
                default=_run_ci_watch,
            ),
        ),
    )


configure_runtime_flags = _register_runtime_flags_callback(
    app=app,
    cli_transport_mode=CliTransportMode,
    apply_runtime_policy_from_env_fn=policy_runtime.apply_runtime_policy_from_env,
    apply_cli_timeout_flag_fn=env_policy.apply_cli_timeout_flag,
    apply_cli_transport_flags_fn=transport_policy.apply_cli_transport_flags,
)


def _cli_timeout_ticks() -> tuple[int, int]:
    budget = deadline_policy.timeout_budget_from_lsp_env(
        default_budget=deadline_policy.DeadlineBudget(
            ticks=_DEFAULT_TIMEOUT_TICKS,
            tick_ns=_DEFAULT_TIMEOUT_TICK_NS,
        )
    )
    return budget.ticks, budget.tick_ns


def _resolve_check_report_path(report: Path | None, *, root: Path) -> Path:
    return path_policy.resolve_report_path(report, root=root)


@contextmanager
def _cli_deadline_scope():
    ticks, tick_ns = _cli_timeout_ticks()
    with deadline_policy.deadline_scope_from_ticks(
        deadline_policy.DeadlineBudget(ticks=ticks, tick_ns=tick_ns),
        gas_limit=int(ticks),
    ):
        yield


CheckArtifactFlags = check_contract.CheckArtifactFlags
CheckPolicyFlags = check_contract.CheckPolicyFlags
DataflowPayloadCommonOptions = check_contract.DataflowPayloadCommonOptions
DataflowFilterBundle = check_contract.DataflowFilterBundle
CheckDeltaOptions = check_contract.CheckDeltaOptions
CheckAuxOperation = check_contract.CheckAuxOperation


@dataclass(frozen=True)
class SnapshotDiffRequest:
    baseline: Path
    current: Path

    def to_payload(self) -> JSONObject:
        return {"baseline": str(self.baseline), "current": str(self.current)}


@dataclass(frozen=True)
class ExecutionPlanRequest:
    requested_operations: list[str]
    inputs: JSONObject
    derived_artifacts: list[str]
    obligations: dict[str, list[str]]
    policy_metadata: dict[str, object]

    def to_payload(self) -> JSONObject:
        return {
            "requested_operations": list(self.requested_operations),
            "inputs": dict(self.inputs),
            "derived_artifacts": list(self.derived_artifacts),
            "obligations": {
                "preconditions": list(self.obligations.get("preconditions") or []),
                "postconditions": list(self.obligations.get("postconditions") or []),
            },
            "policy_metadata": dict(self.policy_metadata),
        }


def _run_sppf_git(
    args: list[str],
    *,
    check_output_fn: Callable[..., str] = subprocess.check_output,
) -> str:
    return check_output_fn(["git", *args], text=True).strip()


def _default_sppf_rev_range(
    *,
    run_sppf_git_fn: Callable[[list[str]], str] = _run_sppf_git,
) -> str:
    try:
        run_sppf_git_fn(["rev-parse", "origin/stage"])
        return "origin/stage..HEAD"
    except Exception:
        return "HEAD~20..HEAD"


def _collect_sppf_commits(
    rev_range: str,
    *,
    check_output_fn: Callable[..., str] = subprocess.check_output,
) -> list[SppfSyncCommitInfo]:
    try:
        raw = check_output_fn(
            [
                "git",
                "log",
                "--format=%H%x1f%s%x1f%B%x1e",
                rev_range,
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise typer.BadParameter(f"git log failed for range {rev_range}: {exc}") from exc

    commits: list[SppfSyncCommitInfo] = []
    for record in raw.split("\x1e"):
        check_deadline()
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) < 3:
            continue
        sha, subject, body = parts[0].strip(), parts[1].strip(), parts[2].strip()
        commits.append(SppfSyncCommitInfo(sha=sha, subject=subject, body=body))
    return commits


def _extract_sppf_issue_ids(text: str) -> set[str]:
    def _canonical(issue_id: str) -> str:
        normalized = issue_id.lstrip("0")
        return normalized or "0"

    issues = set(_canonical(match.group(1)) for match in _SPPF_GH_REF_RE.finditer(text))
    issues.update(_canonical(match.group(1)) for match in _SPPF_KEYWORD_REF_RE.finditer(text))
    return issues


def _normalize_sppf_issue_ids_for_commit(
    commit: SppfSyncCommitInfo,
    issue_ids: set[str],
) -> set[str]:
    normalized = set(issue_ids)
    if "0" not in normalized:
        return normalized
    normalized.discard("0")
    replacement = _SPPF_PLACEHOLDER_ISSUE_BY_COMMIT.get(commit.sha)
    if replacement is not None:
        normalized.add(replacement)
    else:
        normalized.add("0")
    return normalized


def _issue_ids_from_sppf_commits(commits: list[SppfSyncCommitInfo]) -> set[str]:
    issues: set[str] = set()
    for commit in commits:
        check_deadline()
        commit_issue_ids = _extract_sppf_issue_ids(commit.subject)
        commit_issue_ids.update(_extract_sppf_issue_ids(commit.body))
        issues.update(_normalize_sppf_issue_ids_for_commit(commit, commit_issue_ids))
    return issues


def _build_sppf_comment(rev_range: str, commits: list[SppfSyncCommitInfo]) -> str:
    lines = [f"SPPF sync from `{rev_range}`:"]
    for commit in commits:
        check_deadline()
        lines.append(f"- {commit.sha[:8]} {commit.subject}")
    return "\n".join(lines)


def _run_sppf_gh(
    args: list[str],
    *,
    dry_run: bool,
    run_fn: Callable[..., object] = subprocess.run,
) -> None:
    if dry_run:
        typer.echo("DRY RUN: " + " ".join(["gh", *args]))
        return
    run_fn(["gh", *args], check=True)


def _run_sppf_sync(
    *,
    rev_range: str | None,
    comment: bool,
    close: bool,
    label: str | None,
    dry_run: bool,
    default_rev_range_fn: Callable[[], str] = _default_sppf_rev_range,
    collect_sppf_commits_fn: Callable[[str], list[SppfSyncCommitInfo]] = _collect_sppf_commits,
    run_sppf_gh_fn: Callable[[list[str]], None] | None = None,
) -> int:
    resolved_range = rev_range or default_rev_range_fn()
    commits = collect_sppf_commits_fn(resolved_range)
    if not commits:
        typer.echo("No commits in range; nothing to sync.")
        return 0

    issue_ids = sort_once(
        _issue_ids_from_sppf_commits(commits),
        source="gabion.cli.sppf_sync.issue_ids",
    )
    if not issue_ids:
        typer.echo("No issue references found in commit messages.")
        return 0

    summary_comment = _build_sppf_comment(resolved_range, commits)
    gh_runner = run_sppf_gh_fn or (lambda args: _run_sppf_gh(args, dry_run=dry_run))
    for issue_id in issue_ids:
        check_deadline()
        if close:
            gh_runner(["issue", "close", issue_id, "-c", summary_comment])
        elif comment:
            gh_runner(["issue", "comment", issue_id, "-b", summary_comment])
        if label:
            gh_runner(["issue", "edit", issue_id, "--add-label", label])
    return 0


def run_sppf_sync_compat(
    argv: list[str] | None = None,
    *,
    run_sppf_sync_fn: Callable[..., int] = _run_sppf_sync,
) -> int:
    parser = argparse.ArgumentParser(description="Sync SPPF-linked issues from commit messages.")
    parser.add_argument(
        "--range",
        dest="rev_range",
        default=None,
        help="Git revision range (default: origin/stage..HEAD if available).",
    )
    parser.add_argument(
        "--comment",
        action="store_true",
        help="Comment on each referenced issue with commit summary.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close each referenced issue with a summary comment.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Apply a label to each referenced issue.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gh commands without executing.",
    )
    args = parser.parse_args(argv)
    return run_sppf_sync_fn(
        rev_range=args.rev_range,
        comment=args.comment,
        close=args.close,
        label=args.label,
        dry_run=args.dry_run,
    )


def _split_csv_entries(entries: List[str]) -> list[str]:
    check_deadline()
    merged: list[str] = []
    for entry in entries:
        check_deadline()
        merged.extend([part.strip() for part in entry.split(",") if part.strip()])
    return merged


def _split_csv(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items


def _parse_lint_line(line: str) -> dict[str, object] | None:
    match = _LINT_RE.match(line.strip())
    if not match:
        return None
    line_no = int(match.group("line"))
    col_no = int(match.group("col"))
    rest = match.group("rest").strip()
    if not rest:
        return None
    code, _, message = rest.partition(" ")
    return {
        "path": match.group("path"),
        "line": line_no,
        "col": col_no,
        "code": code,
        "message": message,
        "severity": "warning",
    }


def _collect_lint_entries(lines: list[str]) -> list[dict[str, object]]:
    check_deadline()
    entries: list[dict[str, object]] = []
    for line in lines:
        check_deadline()
        parsed = _parse_lint_line(line)
        if parsed is not None:
            entries.append(parsed)
    return entries


def _normalize_output_target(target: str | Path) -> str:
    target_str = str(target)
    if target_str == _STDOUT_ALIAS:
        return _STDOUT_PATH
    return target_str


def _is_stdout_target(target: object) -> bool:
    if target is None:
        return False
    return _normalize_output_target(str(target)) == _STDOUT_PATH


class _TargetStreamRouter:
    def __init__(self, *, max_open_streams: int = 32) -> None:
        self._max_open_streams = max(int(max_open_streams), 1)
        self._streams: OrderedDict[str, io.TextIOWrapper] = OrderedDict()

    def _stream_for_target(
        self,
        target: str,
        *,
        encoding: str,
    ) -> io.TextIOWrapper:
        existing = self._streams.pop(target, None)
        if existing is not None:
            if existing.encoding.lower() != encoding.lower():
                existing.close()
            else:
                self._streams[target] = existing
                return existing
        if len(self._streams) >= self._max_open_streams:
            _, oldest = self._streams.popitem(last=False)
            oldest.close()
        stream = open(target, "w+", encoding=encoding)
        self._streams[target] = stream
        return stream

    def write(
        self,
        *,
        target: str,
        payload: str,
        ensure_trailing_newline: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        text = payload
        if ensure_trailing_newline and not text.endswith("\n"):
            text = text + "\n"
        if target == _STDOUT_PATH:
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        stream = self._stream_for_target(target, encoding=encoding)
        stream.seek(0)
        stream.truncate(0)
        stream.write(text)
        stream.flush()

    def close(self) -> None:
        while self._streams:
            _, stream = self._streams.popitem(last=False)
            stream.close()


_TARGET_STREAM_ROUTER = _TargetStreamRouter()
atexit.register(_TARGET_STREAM_ROUTER.close)


def _write_text_to_target(
    target: str | Path,
    payload: str,
    *,
    ensure_trailing_newline: bool = False,
    encoding: str = "utf-8",
) -> None:
    normalized_target = _normalize_output_target(target)
    _TARGET_STREAM_ROUTER.write(
        target=normalized_target,
        payload=payload,
        ensure_trailing_newline=ensure_trailing_newline,
        encoding=encoding,
    )


def _emit_result_json_to_stdout(*, payload: object) -> None:
    _write_text_to_target(
        _STDOUT_PATH,
        json.dumps(payload, indent=2, sort_keys=False),
        ensure_trailing_newline=True,
    )


def _normalize_optional_output_target(target: object) -> str | None:
    if target is None:
        return None
    text = str(target).strip()
    if not text:
        return None
    return _normalize_output_target(text)


def _write_lint_jsonl(target: str, entries: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(entry, sort_keys=False) for entry in entries)
    _write_text_to_target(
        target,
        payload,
        ensure_trailing_newline=bool(payload),
    )


_write_lint_sarif = cast(
    Callable[[str, list[dict[str, object]]], None],
    partial(
        _write_lint_sarif_impl,
        check_deadline_fn=check_deadline,
        sort_once_fn=sort_once,
        write_text_to_target_fn=_write_text_to_target,
    ),
)


def _emit_lint_outputs(
    lint_lines: list[str],
    *,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
    lint_entries: list[dict[str, object]] | None = None,
) -> None:
    check_deadline()
    if lint:
        for line in lint_lines:
            check_deadline()
            typer.echo(line)
    if lint_jsonl or lint_sarif:
        entries = lint_entries if lint_entries is not None else _collect_lint_entries(lint_lines)
        if lint_jsonl is not None:
            _write_lint_jsonl(str(lint_jsonl), entries)
        if lint_sarif is not None:
            _write_lint_sarif(str(lint_sarif), entries)


def _emit_timeout_profile_artifacts(
    result: Mapping[str, object],
    *,
    root: Path,
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
        render_deadline_profile_markdown(profile) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote deadline profile JSON: {profile_json_path}")
    typer.echo(f"Wrote deadline profile markdown: {profile_md_path}")


_render_timeout_progress_markdown = _render_timeout_progress_markdown_impl


def _build_dataflow_payload_common(
    *,
    options: DataflowPayloadCommonOptions,
) -> JSONObject:
    # dataflow-bundle: filter_bundle
    # dataflow-bundle: deadline_profile
    return check_contract.build_dataflow_payload_common(options=options)


build_check_payload = check_contract.build_check_payload




_check_derived_artifacts = _check_derived_artifacts_impl


build_check_execution_plan_request = cast(
    Callable[..., ExecutionPlanRequest],
    partial(
        _build_check_execution_plan_request_impl,
        check_derived_artifacts_fn=_check_derived_artifacts,
        execution_plan_request_ctor=ExecutionPlanRequest,
        dataflow_command=DATAFLOW_COMMAND,
        check_command=CHECK_COMMAND,
    ),
)
def parse_dataflow_args_or_exit(
    argv: list[str],
    *,
    parser_fn: Callable[[], argparse.ArgumentParser] | None = None,
) -> argparse.Namespace:
    return _parse_dataflow_args_or_exit_impl(argv, parser_fn=parser_fn)


def build_dataflow_payload(opts: argparse.Namespace) -> JSONObject:
    return _build_dataflow_payload_impl(
        opts,
        normalize_optional_output_target_fn=_normalize_optional_output_target,
        build_dataflow_payload_common_fn=_build_dataflow_payload_common,
    )


build_refactor_payload = cast(
    Callable[..., JSONObject],
    partial(
        _build_refactor_payload_impl,
        check_deadline_fn=check_deadline,
    ),
)


def dispatch_command(
    *,
    command: str,
    payload: JSONObject,
    root: Path = Path("."),
    runner: Runner = run_command,
    process_factory: Callable[..., subprocess.Popen] | None = None,
    execution_plan_request: ExecutionPlanRequest | None = None,
    notification_callback: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    return _dispatch_command_impl(
        command=command,
        payload=payload,
        root=root,
        runner=runner,
        process_factory=process_factory,
        execution_plan_request=execution_plan_request,
        notification_callback=notification_callback,
        cli_timeout_ticks_fn=_cli_timeout_ticks,
        normalize_boundary_mapping_once_fn=boundary_order.normalize_boundary_mapping_once,
        apply_boundary_updates_once_fn=boundary_order.apply_boundary_updates_once,
        enforce_boundary_mapping_ordered_fn=boundary_order.enforce_boundary_mapping_ordered,
        command_request_ctor=CommandRequest,
        resolve_command_transport_fn=transport_policy.resolve_command_transport,
        default_lsp_runner=run_command,
        direct_runner=run_command_direct,
        never_fn=never,
    )


run_check = cast(
    Callable[..., JSONObject],
    partial(
        _run_check_impl,
        runner=run_command,
        resolve_check_report_path_fn=_resolve_check_report_path,
        build_check_payload_fn=build_check_payload,
        build_check_execution_plan_request_fn=build_check_execution_plan_request,
        dispatch_command_fn=dispatch_command,
        dataflow_command=DATAFLOW_COMMAND,
    ),
)


def _run_with_timeout_retries(
    *,
    run_once: Callable[[], JSONObject],
    root: Path,
    emit_timeout_profile_artifacts_fn: Callable[..., None] = _emit_timeout_profile_artifacts,
) -> JSONObject:
    with _cli_deadline_scope():
        check_deadline()
        result = run_once()
    if result.get("timeout") is not True:
        return result
    emit_timeout_profile_artifacts_fn(result, root=root)
    raise typer.Exit(code=int(result.get("exit_code", 2)))


def _emit_dataflow_result_outputs(result: JSONObject, opts: argparse.Namespace) -> None:
    return _emit_dataflow_result_outputs_impl(
        result,
        opts,
        cli_deadline_scope_factory=_cli_deadline_scope,
        emit_lint_outputs_fn=_emit_lint_outputs,
        is_stdout_target_fn=_is_stdout_target,
        write_text_to_target_fn=_write_text_to_target,
        emit_result_json_to_stdout_fn=_emit_result_json_to_stdout,
        stdout_path=_STDOUT_PATH,
        check_deadline_fn=check_deadline,
        normalize_dataflow_response_fn=command_orchestrator_primitives._normalize_dataflow_response,
    )


def _param_is_command_line(ctx: typer.Context, param: str) -> bool:
    return ctx.get_parameter_source(param) is ParameterSource.COMMANDLINE


def _raw_profile_unsupported_flags(ctx: typer.Context) -> list[str]:
    strict_only_flags = {
        "emit_test_obsolescence": "--emit-test-obsolescence",
        "emit_test_obsolescence_state": "--emit-test-obsolescence-state",
        "test_obsolescence_state": "--test-obsolescence-state",
        "emit_test_obsolescence_delta": "--emit-test-obsolescence-delta",
        "emit_test_evidence_suggestions": "--emit-test-evidence-suggestions",
        "emit_call_clusters": "--emit-call-clusters",
        "emit_call_cluster_consolidation": "--emit-call-cluster-consolidation",
        "emit_test_annotation_drift": "--emit-test-annotation-drift",
        "emit_semantic_coverage_map": "--emit-semantic-coverage-map",
        "semantic_coverage_mapping": "--semantic-coverage-mapping",
        "test_annotation_drift_state": "--test-annotation-drift-state",
        "emit_test_annotation_drift_delta": "--emit-test-annotation-drift-delta",
        "write_test_annotation_drift_baseline": "--write-test-annotation-drift-baseline",
        "write_test_obsolescence_baseline": "--write-test-obsolescence-baseline",
        "emit_ambiguity_delta": "--emit-ambiguity-delta",
        "emit_ambiguity_state": "--emit-ambiguity-state",
        "ambiguity_state": "--ambiguity-state",
        "write_ambiguity_baseline": "--write-ambiguity-baseline",
        "analysis_tick_limit": "--analysis-tick-limit",
    }
    return [
        flag
        for param, flag in strict_only_flags.items()
        if _param_is_command_line(ctx, param)
    ]


_check_raw_profile_args = cast(
    Callable[..., list[str]],
    partial(
        _check_raw_profile_args_impl,
        param_is_command_line_fn=_param_is_command_line,
        deadline_loop_iter_fn=deadline_loop_iter,
        dataflow_filter_bundle_ctor=DataflowFilterBundle,
    ),
)


_run_check_raw_profile = cast(
    Callable[..., None],
    partial(
        _run_check_raw_profile_impl,
        raw_profile_unsupported_flags_fn=_raw_profile_unsupported_flags,
        check_raw_profile_args_fn=_check_raw_profile_args,
        default_run_dataflow_raw_argv_fn=lambda argv: _run_dataflow_raw_argv(argv),
    ),
)


def _nonzero_exit_causes(result: JSONObject) -> list[str]:
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


def _emit_nonzero_exit_causes(result: JSONObject) -> None:
    exit_code = int(result.get("exit_code", 0) or 0)
    if exit_code == 0:
        return
    causes = "; ".join(_nonzero_exit_causes(result))
    typer.echo(f"Non-zero exit ({exit_code}) cause(s): {causes}", err=True)


def _emit_resume_state_startup_line(
    *,
    checkpoint_path: str,
    status: str,
    reused_files: int | None,
    total_files: int | None,
) -> None:
    reused_display = "unknown"
    if isinstance(reused_files, int) and isinstance(total_files, int):
        reused_display = f"{int(reused_files)}/{int(total_files)}"
    typer.echo(
        "resume state detected... "
        f"path={checkpoint_path or '<none>'} "
        f"status={status or 'unknown'} "
        f"reused_files={reused_display}"
    )


def _phase_timeline_header_columns() -> list[str]:
    return progress_timeline.phase_timeline_header_columns()


def _phase_timeline_header_block() -> str:
    return progress_timeline.phase_timeline_header_block()


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, object] | None,
) -> str:
    return progress_timeline.phase_progress_dimensions_summary(phase_progress_v2)


def _phase_timeline_row_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> str:
    return progress_timeline.phase_timeline_row_from_phase_progress(phase_progress)


def _phase_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    return progress_timeline.phase_timeline_from_progress_notification(notification)


def _phase_progress_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    payload = progress_timeline.phase_progress_from_progress_notification(notification)
    if isinstance(payload, Mapping):
        return {str(key): payload[key] for key in payload}
    return None


def _emit_phase_progress_line(phase_progress: Mapping[str, object]) -> None:
    phase = str(phase_progress.get("phase", "") or "")
    if not phase:
        return
    analysis_state = str(phase_progress.get("analysis_state", "") or "")
    classification = str(phase_progress.get("classification", "") or "")
    work_done = phase_progress.get("work_done")
    work_total = phase_progress.get("work_total")
    completed_files = phase_progress.get("completed_files")
    remaining_files = phase_progress.get("remaining_files")
    total_files = phase_progress.get("total_files")
    done = bool(phase_progress.get("done", False))
    fragments = [f"phase={phase}"]
    if analysis_state:
        fragments.append(f"analysis_state={analysis_state}")
    if classification:
        fragments.append(f"classification={classification}")
    if isinstance(work_done, int) and isinstance(work_total, int):
        fragments.append(f"work={work_done}/{work_total}")
    if (
        isinstance(completed_files, int)
        and isinstance(remaining_files, int)
        and isinstance(total_files, int)
    ):
        fragments.append(
            f"files={completed_files}/{total_files} remaining={remaining_files}"
        )
    prefix = "progress done" if done else "progress"
    typer.echo(f"{prefix} {' '.join(fragments)}")


def _emit_phase_timeline_progress(*, header: str | None, row: str) -> None:
    if isinstance(header, str) and header:
        typer.echo(header)
    typer.echo(row)


def _emit_analysis_resume_summary(result: JSONObject) -> None:
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


def _context_run_dataflow_raw_argv(ctx: typer.Context) -> Callable[[list[str]], None]:
    return _context_cli_deps(ctx).run_dataflow_raw_argv_fn


def _context_run_check(ctx: typer.Context) -> Callable[..., JSONObject]:
    return _context_cli_deps(ctx).run_check_fn


def _context_run_ci_watch(
    ctx: typer.Context,
) -> CliRunCiWatchFn:
    return _context_cli_deps(ctx).run_ci_watch_fn


def _run_dataflow_raw_argv(
    argv: list[str],
    *,
    runner: Runner | None = None,
) -> None:
    opts = parse_dataflow_args_or_exit(argv)
    payload = build_dataflow_payload(opts)
    resolved_runner = runner or run_command
    timeline_header_emitted = False
    last_phase_progress_signature: tuple[object, ...] | None = None
    last_phase_event_seq: int | None = None

    def _on_notification(notification: JSONObject) -> None:
        nonlocal timeline_header_emitted
        nonlocal last_phase_progress_signature
        nonlocal last_phase_event_seq
        phase_progress = _phase_progress_from_progress_notification(notification)
        if not isinstance(phase_progress, Mapping):
            return
        event_seq = phase_progress.get("event_seq")
        if isinstance(event_seq, int):
            if last_phase_event_seq == event_seq:
                return
            last_phase_event_seq = event_seq
        signature = progress_timeline.phase_progress_signature(phase_progress)
        if signature == last_phase_progress_signature:
            return
        last_phase_progress_signature = signature
        timeline_update = progress_timeline.phase_timeline_from_phase_progress(
            phase_progress
        )
        row = str(timeline_update.get("row") or "")
        header_value = timeline_update.get("header")
        header = (
            header_value
            if not timeline_header_emitted and isinstance(header_value, str) and header_value
            else None
        )
        _emit_phase_timeline_progress(header=header, row=row)
        if header is not None:
            timeline_header_emitted = True

    result = _run_with_timeout_retries(
        run_once=lambda: dispatch_command(
            command=DATAFLOW_COMMAND,
            payload=payload,
            root=Path(opts.root),
            runner=resolved_runner,
            notification_callback=_on_notification,
        ),
        root=Path(opts.root),
    )
    _emit_dataflow_result_outputs(result, opts)
    _emit_analysis_resume_summary(result)
    _emit_nonzero_exit_causes(result)
    raise typer.Exit(code=int(result.get("exit_code", 0)))


def _default_check_artifact_flags() -> CheckArtifactFlags:
    return CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=False,
        emit_semantic_coverage_map=False,
    )


def _default_check_delta_options() -> CheckDeltaOptions:
    return CheckDeltaOptions(
        obsolescence_mode=check_contract.CheckAuxMode(kind="off"),
        annotation_drift_mode=check_contract.CheckAuxMode(kind="off"),
        ambiguity_mode=check_contract.CheckAuxMode(kind="off"),
        semantic_coverage_mapping=None,
    )


def _check_help_or_exit(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=2)


def _check_gate_policy(gate: CheckGateMode) -> tuple[bool, bool]:
    if gate is CheckGateMode.all:
        return True, True
    if gate is CheckGateMode.none:
        return False, False
    if gate is CheckGateMode.violations:
        return True, False
    if gate is CheckGateMode.type_ambiguities:
        return False, True
    never("invalid check gate mode", gate=str(gate))
    return False, False  # pragma: no cover


def _check_lint_mode(
    *,
    lint_mode: CheckLintMode,
    lint_jsonl_out: Path | None,
    lint_sarif_out: Path | None,
) -> tuple[bool, bool]:
    line_enabled = lint_mode in {
        CheckLintMode.line,
        CheckLintMode.line_jsonl,
        CheckLintMode.line_sarif,
        CheckLintMode.all,
    }
    jsonl_enabled = lint_mode in {
        CheckLintMode.jsonl,
        CheckLintMode.line_jsonl,
        CheckLintMode.jsonl_sarif,
        CheckLintMode.all,
    }
    sarif_enabled = lint_mode in {
        CheckLintMode.sarif,
        CheckLintMode.line_sarif,
        CheckLintMode.jsonl_sarif,
        CheckLintMode.all,
    }
    if jsonl_enabled and lint_jsonl_out is None:
        raise typer.BadParameter(
            "--lint-jsonl-out is required when --lint includes jsonl output."
        )
    if sarif_enabled and lint_sarif_out is None:
        raise typer.BadParameter(
            "--lint-sarif-out is required when --lint includes sarif output."
        )
    if not jsonl_enabled and lint_jsonl_out is not None:
        raise typer.BadParameter(
            "--lint-jsonl-out is only valid when --lint includes jsonl."
        )
    if not sarif_enabled and lint_sarif_out is not None:
        raise typer.BadParameter(
            "--lint-sarif-out is only valid when --lint includes sarif."
        )
    lint_enabled = lint_mode is not CheckLintMode.none
    return lint_enabled, line_enabled


_build_status_watch_options = cast(
    Callable[..., tooling_ci_watch.StatusWatchOptions | None],
    partial(
        _build_status_watch_options_impl,
        default_status_watch_artifact_root=_DEFAULT_STATUS_WATCH_ARTIFACT_ROOT,
        status_watch_options_ctor=tooling_ci_watch.StatusWatchOptions,
    ),
)


def _emit_status_watch_outcome(
    *,
    result: tooling_ci_watch.StatusWatchResult,
    options: tooling_ci_watch.StatusWatchOptions,
) -> None:
    line = f"status-watch run_id={result.run_id}"
    if options.summary_json is not None:
        line = f"{line} summary={options.summary_json}"
    if result.collection is not None:
        line = f"{line} failure_bundle={result.collection.run_root}"
    typer.echo(line)


def _run_check_delta_gates(
    *,
    gate_specs: tuple[tool_specs.ToolSpec, ...] | None = None,
) -> int:
    specs = (
        tuple(tool_specs.dataflow_stage_gate_specs())
        if gate_specs is None
        else tuple(gate_specs)
    )
    with _cli_deadline_scope():
        tooling_delta_advisory.telemetry_main()
        return next(
            (
                gate_exit
                for gate_exit in (
                    int(spec.run())
                    for spec in deadline_loop_iter(specs)
                )
                if gate_exit != 0
            ),
            0,
        )


def _run_ci_watch(
    options: tooling_ci_watch.StatusWatchOptions,
) -> tooling_ci_watch.StatusWatchResult:
    return tooling_ci_watch.run_watch(options=options)


_run_check_command = cast(
    Callable[..., None],
    partial(
        _run_check_command_impl,
        check_gate_policy_fn=_check_gate_policy,
        check_lint_mode_fn=_check_lint_mode,
        context_cli_deps_fn=_context_cli_deps,
        phase_progress_from_progress_notification_fn=_phase_progress_from_progress_notification,
        phase_progress_signature_fn=progress_timeline.phase_progress_signature,
        phase_timeline_from_phase_progress_fn=progress_timeline.phase_timeline_from_phase_progress,
        emit_phase_timeline_progress_fn=_emit_phase_timeline_progress,
        run_with_timeout_retries_fn=_run_with_timeout_retries,
        cli_deadline_scope_factory=_cli_deadline_scope,
        emit_lint_outputs_fn=_emit_lint_outputs,
        emit_analysis_resume_summary_fn=_emit_analysis_resume_summary,
        emit_nonzero_exit_causes_fn=_emit_nonzero_exit_causes,
        emit_status_watch_outcome_fn=_emit_status_watch_outcome,
        check_policy_flags_ctor=CheckPolicyFlags,
        path_ctor=Path,
    ),
)


check_delta_bundle = _register_check_delta_bundle_command(
    check_app=check_app,
    check_strictness_mode=CheckStrictnessMode,
    check_gate_mode=CheckGateMode,
    check_lint_mode=CheckLintMode,
    run_check_command_fn=_run_check_command,
    dataflow_filter_bundle_ctor=DataflowFilterBundle,
    delta_bundle_artifact_flags_fn=check_contract.delta_bundle_artifact_flags,
    delta_bundle_delta_options_fn=check_contract.delta_bundle_delta_options,
)


@check_app.command("delta-gates")
def check_delta_gates(ctx: typer.Context) -> None:
    deps = _context_cli_deps(ctx)
    raise typer.Exit(code=deps.run_check_delta_gates_fn())


_run_check_aux_operation = cast(
    Callable[..., None],
    partial(
        _run_check_aux_operation_impl,
        run_check_command_fn=_run_check_command,
        default_check_artifact_flags_fn=_default_check_artifact_flags,
        default_check_delta_options_fn=_default_check_delta_options,
        dataflow_filter_bundle_ctor=DataflowFilterBundle,
        gate_none=CheckGateMode.none,
        lint_mode_none=CheckLintMode.none,
    ),
)


check_group = _register_check_group_callback(
    check_app=check_app,
    check_help_or_exit_fn=_check_help_or_exit,
)


@check_obsolescence_app.callback()
def check_obsolescence_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_annotation_drift_app.callback()
def check_annotation_drift_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_ambiguity_app.callback()
def check_ambiguity_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_taint_app.callback()
def check_taint_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_app.command("raw", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def check_raw(ctx: typer.Context) -> None:
    raw_args = list(ctx.args)
    if raw_args and raw_args[0] == "--":
        raw_args = raw_args[1:]
    if not raw_args:
        raise typer.BadParameter("Usage: gabion check raw -- [raw-args...]")
    run_raw = _context_run_dataflow_raw_argv(ctx)
    run_raw(raw_args)


check_run = _register_check_run_command(
    check_app=check_app,
    check_strictness_mode=CheckStrictnessMode,
    check_baseline_mode=CheckBaselineMode,
    check_gate_mode=CheckGateMode,
    check_lint_mode=CheckLintMode,
    dataflow_filter_bundle_ctor=DataflowFilterBundle,
    build_status_watch_options_fn=_build_status_watch_options,
    run_check_command_fn=_run_check_command,
    default_check_artifact_flags_fn=_default_check_artifact_flags,
    default_check_delta_options_fn=_default_check_delta_options,
)


@check_obsolescence_app.command("report")
def check_obsolescence_report(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="obsolescence",
        action="report",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_obsolescence_app.command("state")
def check_obsolescence_state(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="obsolescence",
        action="state",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=None,
        out_json=out_json,
        out_md=None,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_obsolescence_app.command("delta")
def check_obsolescence_delta(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="obsolescence",
        action="delta",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_obsolescence_app.command("baseline-write")
def check_obsolescence_baseline_write(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="obsolescence",
        action="baseline-write",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_annotation_drift_app.command("report")
def check_annotation_drift_report(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="annotation-drift",
        action="report",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_annotation_drift_app.command("state")
def check_annotation_drift_state(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="annotation-drift",
        action="state",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=None,
        out_json=out_json,
        out_md=None,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_annotation_drift_app.command("delta")
def check_annotation_drift_delta(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="annotation-drift",
        action="delta",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_annotation_drift_app.command("baseline-write")
def check_annotation_drift_baseline_write(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="annotation-drift",
        action="baseline-write",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_ambiguity_app.command("state")
def check_ambiguity_state(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="ambiguity",
        action="state",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=None,
        out_json=out_json,
        out_md=None,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_ambiguity_app.command("delta")
def check_ambiguity_delta(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="ambiguity",
        action="delta",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_ambiguity_app.command("baseline-write")
def check_ambiguity_baseline_write(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="ambiguity",
        action="baseline-write",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_taint_app.command("state")
def check_taint_state(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="taint",
        action="state",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=None,
        out_json=out_json,
        out_md=None,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_taint_app.command("delta")
def check_taint_delta(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="taint",
        action="delta",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_taint_app.command("baseline-write")
def check_taint_baseline_write(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Path = typer.Option(..., "--baseline"),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    out_md: Optional[Path] = typer.Option(None, "--out-md"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="taint",
        action="baseline-write",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=baseline,
        state_in=state_in,
        out_json=out_json,
        out_md=out_md,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


@check_taint_app.command("lifecycle")
def check_taint_lifecycle(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    state_in: Optional[Path] = typer.Option(None, "--state-in"),
    out_json: Optional[Path] = typer.Option(None, "--out-json"),
    report: Optional[Path] = typer.Option(None, "--report"),
    decision_snapshot: Optional[Path] = typer.Option(None, "--decision-snapshot"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_state_json: Optional[Path] = typer.Option(None, "--aspf-state-json"),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
) -> None:
    _run_check_aux_operation(
        ctx=ctx,
        domain="taint",
        action="lifecycle",
        paths=paths,
        root=root,
        config=config,
        strictness=strictness,
        allow_external=allow_external,
        baseline=None,
        state_in=state_in,
        out_json=out_json,
        out_md=None,
        report=report,
        decision_snapshot=decision_snapshot,
        analysis_budget_checks=analysis_budget_checks,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
    )


def dataflow_cli_parser() -> argparse.ArgumentParser:
    return _build_dataflow_cli_parser()


def _run_governance_runner(
    *,
    runner_name: str,
    runner: Callable[[list[str] | None], int],
    args: list[str],
) -> int:
    try:
        return int(runner(args))
    except Exception as exc:
        typer.secho(
            f"governance command failed ({runner_name}): {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 1


_restore_aspf_state_from_github_artifacts = (
    _github_artifact_restore._restore_aspf_state_from_github_artifacts
)
_NoRedirectHandler = _github_artifact_restore._NoRedirectHandler
_download_artifact_archive_bytes = _github_artifact_restore._download_artifact_archive_bytes
_state_requires_chunk_artifacts = _github_artifact_restore._state_requires_chunk_artifacts


def _run_docflow_audit(
    *,
    root: Path,
    fail_on_violations: bool,
    sppf_gh_ref_mode: str = "required",
    extra_path: list[str] | None = None,
) -> int:
    args = ["--root", str(root)]
    for entry in extra_path or []:
        args.extend(["--extra-path", entry])
    if fail_on_violations:
        args.append("--fail-on-violations")
    args.extend(["--sppf-gh-ref-mode", sppf_gh_ref_mode])

    status = _run_governance_runner(
        runner_name="run_docflow_cli",
        runner=tooling_governance_audit.run_docflow_cli,
        args=args,
    )
    if status != 0:
        return status
    try:
        return int(tooling_governance_audit.run_sppf_graph_cli([]))
    except Exception as exc:
        typer.secho(
            f"docflow: sppf-graph failed: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 1


def _context_run_sppf_sync(ctx: typer.Context) -> Callable[..., int]:
    return _context_cli_deps(ctx).run_sppf_sync_fn


@app.command("sppf-sync")
def sppf_sync(
    ctx: typer.Context,
    rev_range: Optional[str] = typer.Option(
        None,
        "--range",
        help="Git revision range (default: origin/stage..HEAD if available).",
    ),
    comment: bool = typer.Option(
        False,
        "--comment",
        help="Comment on each referenced issue with commit summary.",
    ),
    close: bool = typer.Option(
        False,
        "--close",
        help="Close each referenced issue with a summary comment.",
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Apply a label to each referenced issue.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print gh commands without executing.",
    ),
) -> None:
    """Sync SPPF checklist-linked GitHub issues from commit messages."""
    deps = _context_cli_deps(ctx)
    with _cli_deadline_scope():
        try:
            exit_code = deps.run_sppf_sync_fn(
                rev_range=rev_range,
                comment=comment,
                close=close,
                label=label,
                dry_run=dry_run,
            )
        except (subprocess.CalledProcessError, typer.BadParameter) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc
        raise typer.Exit(code=exit_code)


@app.command("docflow")
def docflow(
    root: Path = typer.Option(Path("."), "--root"),
    fail_on_violations: bool = typer.Option(
        True, "--fail-on-violations/--no-fail-on-violations"
    ),
    sppf_gh_ref_mode: str = typer.Option(
        "required",
        "--sppf-gh-ref-mode",
        help="SPPF GH-reference enforcement mode (required|advisory).",
    ),
    extra_path: list[str] = typer.Option([], "--extra-path"),
) -> None:
    """Run the docflow audit (governance docs only)."""
    exit_code = _run_docflow_audit(
        root=root,
        fail_on_violations=fail_on_violations,
        sppf_gh_ref_mode=sppf_gh_ref_mode,
        extra_path=extra_path,
    )
    raise typer.Exit(code=exit_code)


@app.command("sppf-graph")
def sppf_graph(
    root: Path = typer.Option(Path("."), "--root"),
    json_output: Path = typer.Option(Path("artifacts/sppf_dependency_graph.json"), "--json-output"),
    dot_output: Path | None = typer.Option(None, "--dot-output"),
    issues_json: Path | None = typer.Option(None, "--issues-json"),
) -> None:
    """Emit SPPF dependency graph artifacts."""
    args = ["--root", str(root), "--json-output", str(json_output)]
    if dot_output is not None:
        args.extend(["--dot-output", str(dot_output)])
    if issues_json is not None:
        args.extend(["--issues-json", str(issues_json)])
    exit_code = _run_governance_runner(
        runner_name="run_sppf_graph_cli",
        runner=tooling_governance_audit.run_sppf_graph_cli,
        args=args,
    )
    raise typer.Exit(code=exit_code)


@app.command("status-consistency")
def status_consistency(
    root: Path = typer.Option(Path("."), "--root"),
    extra_path: list[str] = typer.Option([], "--extra-path"),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
    json_output: Path = typer.Option(Path("artifacts/out/status_consistency.json"), "--json-output"),
    md_output: Path = typer.Option(Path("artifacts/audit_reports/status_consistency.md"), "--md-output"),
) -> None:
    """Check SPPF checklist status consistency across in/ + influence index metadata."""
    args = ["--root", str(root), "--json-output", str(json_output), "--md-output", str(md_output)]
    for entry in extra_path:
        args.extend(["--extra-path", entry])
    if fail_on_violations:
        args.append("--fail-on-violations")
    exit_code = _run_governance_runner(
        runner_name="run_status_consistency_cli",
        runner=tooling_governance_audit.run_status_consistency_cli,
        args=args,
    )
    raise typer.Exit(code=exit_code)


@app.command("decision-tiers")
def decision_tiers(
    root: Path = typer.Option(Path("."), "--root"),
    lint: Path | None = typer.Option(None, "--lint"),
    format: str = typer.Option("toml", "--format"),
) -> None:
    """Extract decision-tier candidates from lint output."""
    args = ["--root", str(root), "--format", format]
    if lint is not None:
        args.extend(["--lint", str(lint)])
    exit_code = _run_governance_runner(
        runner_name="run_decision_tiers_cli",
        runner=tooling_governance_audit.run_decision_tiers_cli,
        args=args,
    )
    raise typer.Exit(code=exit_code)


@app.command("consolidation")
def consolidation(
    root: Path = typer.Option(Path("."), "--root"),
    decision: Path | None = typer.Option(None, "--decision"),
    lint: Path | None = typer.Option(None, "--lint"),
    output: Path | None = typer.Option(None, "--output"),
    json_output: Path | None = typer.Option(None, "--json-output"),
) -> None:
    """Generate consolidation audit report artifacts."""
    args = ["--root", str(root)]
    if decision is not None:
        args.extend(["--decision", str(decision)])
    if lint is not None:
        args.extend(["--lint", str(lint)])
    if output is not None:
        args.extend(["--output", str(output)])
    if json_output is not None:
        args.extend(["--json-output", str(json_output)])
    exit_code = _run_governance_runner(
        runner_name="run_consolidation_cli",
        runner=tooling_governance_audit.run_consolidation_cli,
        args=args,
    )
    raise typer.Exit(code=exit_code)


@app.command("lint-summary")
def lint_summary(
    lint: Path | None = typer.Option(None, "--lint"),
    root: Path = typer.Option(Path("."), "--root"),
    json_mode: bool = typer.Option(False, "--json/--no-json"),
    top: int = typer.Option(10, "--top"),
) -> None:
    """Summarize lint output."""
    args = ["--root", str(root), "--top", str(int(top))]
    if lint is not None:
        args.extend(["--lint", str(lint)])
    if json_mode:
        args.append("--json")
    exit_code = _run_governance_runner(
        runner_name="run_lint_summary_cli",
        runner=tooling_governance_audit.run_lint_summary_cli,
        args=args,
    )
    raise typer.Exit(code=exit_code)


_run_synth = cast(
    Callable[..., tuple[JSONObject, dict[str, Path], Path | None]],
    partial(
        _run_synth_impl,
        dispatch_command_fn=dispatch_command,
        check_deadline_fn=check_deadline,
        dataflow_command=DATAFLOW_COMMAND,
    ),
)


@app.command("synthesis-plan")
def synthesis_plan(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing bundles and synthesis settings."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write synthesis plan JSON to this path."
    ),
) -> None:
    """Generate a synthesis plan from a JSON payload (prototype)."""
    with _cli_deadline_scope():
        _run_synthesis_plan(input_path=input_path, output_path=output_path)


def _run_synthesis_plan(
    *,
    input_path: Optional[Path],
    output_path: Optional[Path],
    runner: Runner = run_command,
) -> None:
    """Generate a synthesis plan from a JSON payload (prototype)."""
    payload: JSONObject = {}
    if input_path is not None:
        try:
            loaded = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(loaded, dict):
            raise typer.BadParameter("Synthesis payload must be a JSON object.")
        payload = loaded
    result = dispatch_command(
        command=SYNTHESIS_COMMAND,
        payload=payload,
        root=Path("."),
        runner=runner,
    )
    normalized = SynthesisPlanResponseDTO.model_validate(result).model_dump()
    output = json.dumps(normalized, indent=2, sort_keys=False)
    if output_path is None:
        typer.echo(output)
    else:
        _write_text_to_target(output_path, output)


def _emit_synth_outputs(
    *,
    paths_out: dict[str, Path],
    timestamp: Path | None,
    refactor_plan: bool,
) -> None:
    if timestamp:
        typer.echo(f"Snapshot: {paths_out['output_root']}")
    typer.echo(f"- {paths_out['report']}")
    typer.echo(f"- {paths_out['dot']}")
    typer.echo(f"- {paths_out['plan']}")
    typer.echo(f"- {paths_out['protocol']}")
    if paths_out["fingerprint_synth"].exists():
        typer.echo(f"- {paths_out['fingerprint_synth']}")
    if paths_out["fingerprint_provenance"].exists():
        typer.echo(f"- {paths_out['fingerprint_provenance']}")
    if paths_out["fingerprint_coherence"].exists():
        typer.echo(f"- {paths_out['fingerprint_coherence']}")
    if paths_out["fingerprint_rewrite_plans"].exists():
        typer.echo(f"- {paths_out['fingerprint_rewrite_plans']}")
    if paths_out["fingerprint_exception_obligations"].exists():
        typer.echo(f"- {paths_out['fingerprint_exception_obligations']}")
    if paths_out["fingerprint_handledness"].exists():
        typer.echo(f"- {paths_out['fingerprint_handledness']}")
    if refactor_plan:
        typer.echo(f"- {paths_out['refactor']}")


synth = _register_synth_command(
    app=app,
    dataflow_filter_bundle_ctor=DataflowFilterBundle,
    cli_deadline_scope_factory=_cli_deadline_scope,
    run_synth_fn=_run_synth,
    emit_synth_outputs_fn=_emit_synth_outputs,
)


def _run_snapshot_diff_command(
    *,
    command: str,
    request: SnapshotDiffRequest,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload = request.to_payload()
    resolved_runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    return dispatch_command(
        command=command,
        payload=payload,
        root=root_path,
        runner=resolved_runner,
    )


def run_structure_diff(
    *,
    request: SnapshotDiffRequest,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    return _run_snapshot_diff_command(
        command=STRUCTURE_DIFF_COMMAND,
        request=request,
        root=root,
        runner=runner,
    )


def run_decision_diff(
    *,
    request: SnapshotDiffRequest,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    return _run_snapshot_diff_command(
        command=DECISION_DIFF_COMMAND,
        request=request,
        root=root,
        runner=runner,
    )


def run_structure_reuse(
    *,
    snapshot: Path,
    min_count: int = 2,
    lemma_stubs: Path | None = None,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload = {"snapshot": str(snapshot), "min_count": int(min_count)}
    if lemma_stubs is not None:
        payload["lemma_stubs"] = str(lemma_stubs)
    runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    return dispatch_command(
        command=STRUCTURE_REUSE_COMMAND,
        payload=payload,
        root=root_path,
        runner=runner,
    )




def run_lsp_parity_gate(
    *,
    commands: list[str] | None = None,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload: JSONObject = {}
    if commands is not None:
        payload["commands"] = list(commands)
    resolved_runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    payload["root"] = str(root_path)
    return dispatch_command(
        command=LSP_PARITY_GATE_COMMAND,
        payload=payload,
        root=root_path,
        runner=resolved_runner,
    )



def run_impact_query(
    *,
    changes: list[str],
    git_diff: str | None,
    max_call_depth: int | None,
    confidence_threshold: float | None,
    root: Path | None = None,
    runner: Runner | None = None,
) -> JSONObject:
    payload: JSONObject = {"changes": list(changes)}
    if git_diff is not None:
        payload["git_diff"] = git_diff
    if max_call_depth is not None:
        payload["max_call_depth"] = int(max_call_depth)
    if confidence_threshold is not None:
        payload["confidence_threshold"] = float(confidence_threshold)
    runner = runner or DEFAULT_RUNNER
    root_path = root or Path(".")
    payload["root"] = str(root_path)
    return dispatch_command(
        command=IMPACT_COMMAND,
        payload=payload,
        root=root_path,
        runner=runner,
    )


def _emit_impact(result: JSONObject, *, json_output: bool) -> None:
    check_deadline()
    errors = result.get("errors")
    exit_code = int(result.get("exit_code", 0))
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=False))
    else:
        must = result.get("must_run_tests") or []
        likely = result.get("likely_run_tests") or []
        docs = result.get("impacted_docs") or []
        typer.echo("must-run tests:")
        if must:
            for entry in must:
                check_deadline()
                typer.echo(f"- {entry.get('id')} (depth={entry.get('depth')}, confidence={entry.get('confidence')})")
        else:
            typer.echo("- (none)")
        typer.echo("likely-run tests:")
        if likely:
            for entry in likely:
                check_deadline()
                typer.echo(f"- {entry.get('id')} (depth={entry.get('depth')}, confidence={entry.get('confidence')})")
        else:
            typer.echo("- (none)")
        typer.echo("impacted docs sections:")
        if docs:
            for entry in docs:
                check_deadline()
                symbols = ",".join(str(s) for s in entry.get("symbols", []))
                typer.echo(f"- {entry.get('path')}#{entry.get('section')} [{symbols}]")
        else:
            typer.echo("- (none)")
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _invoke_argparse_command(
    main_fn: Callable[[list[str] | None], int],
    argv: list[str],
) -> int:
    return _invoke_argparse_command_impl(main_fn, argv)


_TOOLING_NO_ARG_RUNNERS: dict[str, Callable[[], int]] = {
    "delta-advisory-telemetry": tooling_delta_advisory.telemetry_main,
    "docflow-delta-emit": tooling_docflow_delta_emit.main,
}
_TOOLING_ARGV_RUNNERS: dict[str, Callable[[list[str] | None], int]] = {
    "ci-watch": tooling_ci_watch.main,
    "impact-select-tests": tooling_impact_select_tests.main,
    "run-dataflow-stage": tooling_run_dataflow_stage.main,
    "ambiguity-contract-gate": tooling_ambiguity_contract_policy_check.main,
    "normative-symdiff": tooling_normative_symdiff.main,
}


@contextmanager
def _tooling_runner_override(
    *,
    no_arg: Mapping[str, Callable[[], int]] | None = None,
    with_argv: Mapping[str, Callable[[list[str] | None], int]] | None = None,
) -> Generator[None, None, None]:
    with _tooling_runner_override_impl(
        no_arg_runners=_TOOLING_NO_ARG_RUNNERS,
        with_argv_runners=_TOOLING_ARGV_RUNNERS,
        no_arg=no_arg,
        with_argv=with_argv,
    ):
        yield


def _run_tooling_no_arg(command_name: str) -> int:
    return _run_tooling_no_arg_impl(
        command_name=command_name,
        no_arg_runners=_TOOLING_NO_ARG_RUNNERS,
        cli_deadline_scope_factory=_cli_deadline_scope,
    )


def _run_tooling_with_argv(command_name: str, argv: list[str]) -> int:
    return _run_tooling_with_argv_impl(
        command_name=command_name,
        argv=argv,
        with_argv_runners=_TOOLING_ARGV_RUNNERS,
        cli_deadline_scope_factory=_cli_deadline_scope,
    )


@app.command("delta-state-emit", hidden=True)
def removed_delta_state_emit() -> None:
    raise typer.BadParameter(
        "Removed command: delta-state-emit. Use `gabion check delta-bundle`."
    )


@app.command("delta-triplets", hidden=True)
def removed_delta_triplets() -> None:
    raise typer.BadParameter(
        "Removed command: delta-triplets. Use `gabion check delta-gates`."
    )


ci_watch = _register_ci_watch_command(
    app=app,
    default_status_watch_artifact_root=_DEFAULT_STATUS_WATCH_ARTIFACT_ROOT,
    status_watch_options_ctor=tooling_ci_watch.StatusWatchOptions,
    run_tooling_with_argv_fn=_run_tooling_with_argv,
)

_tooling_passthrough_commands = _register_tooling_passthrough_commands(
    app=app,
    run_tooling_no_arg_fn=_run_tooling_no_arg,
    run_tooling_with_argv_fn=_run_tooling_with_argv,
)
delta_advisory_telemetry = _tooling_passthrough_commands["delta_advisory_telemetry"]
docflow_delta_emit = _tooling_passthrough_commands["docflow_delta_emit"]
impact_select_tests = _tooling_passthrough_commands["impact_select_tests"]
run_dataflow_stage = _tooling_passthrough_commands["run_dataflow_stage"]
ambiguity_contract_gate = _tooling_passthrough_commands["ambiguity_contract_gate"]
normative_symdiff = _tooling_passthrough_commands["normative_symdiff"]


@app.command("lsp-parity-gate")
def lsp_parity_gate(
    command: Optional[List[str]] = typer.Option(
        None,
        "--command",
        help="Command ID to validate (repeatable). Defaults to all governed commands.",
    ),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Validate command maturity policy and LSP/direct parity contracts."""
    with _cli_deadline_scope():
        result = run_lsp_parity_gate(commands=list(command or []) or None, root=root)
        normalized = LspParityGateResponseDTO.model_validate(result).model_dump()
        typer.echo(json.dumps(normalized, indent=2, sort_keys=False))
        if int(normalized.get("exit_code", 0)) != 0:
            raise typer.Exit(code=int(normalized["exit_code"]))



@app.command("impact")
def impact(
    change: Optional[List[str]] = typer.Option(
        None,
        "--change",
        help="Changed span as path:start-end (repeatable).",
    ),
    diff: Optional[Path] = typer.Option(
        None,
        "--git-diff",
        help="Path to unified git diff; use '-' for stdin.",
    ),
    max_call_depth: Optional[int] = typer.Option(None, "--max-call-depth"),
    confidence_threshold: Optional[float] = typer.Option(None, "--confidence-threshold"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Query reverse impact from changed spans to tests/docs."""
    with _cli_deadline_scope():
        changes = list(change or [])
        diff_text: str | None = None
        if diff is not None:
            if str(diff) == "-":
                diff_text = sys.stdin.read()
            else:
                diff_text = diff.read_text(encoding="utf-8")
        result = run_impact_query(
            changes=changes,
            git_diff=diff_text,
            max_call_depth=max_call_depth,
            confidence_threshold=confidence_threshold,
            root=root,
        )
        _emit_impact(result, json_output=json_output)
def _emit_structure_diff(result: JSONObject) -> None:
    check_deadline()
    normalized = StructureDiffResponseDTO.model_validate(result).model_dump()
    errors = normalized.get("errors")
    exit_code = int(normalized.get("exit_code", 0))
    typer.echo(json.dumps(normalized, indent=2, sort_keys=False))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _emit_decision_diff(result: JSONObject) -> None:
    check_deadline()
    normalized = DecisionDiffResponseDTO.model_validate(result).model_dump()
    errors = normalized.get("errors")
    exit_code = int(normalized.get("exit_code", 0))
    typer.echo(json.dumps(normalized, indent=2, sort_keys=False))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


def _emit_structure_reuse(result: JSONObject) -> None:
    check_deadline()
    normalized = StructureReuseResponseDTO.model_validate(result).model_dump()
    errors = normalized.get("errors")
    exit_code = int(normalized.get("exit_code", 0))
    typer.echo(json.dumps(normalized, indent=2, sort_keys=False))
    if errors:
        for error in errors:
            check_deadline()
            typer.secho(str(error), err=True, fg=typer.colors.RED)
    if exit_code:
        raise typer.Exit(code=exit_code)


@app.command("structure-diff")
def structure_diff(
    baseline: Path = typer.Option(..., "--baseline"),
    current: Path = typer.Option(..., "--current"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Compare two structure snapshots and emit a JSON diff."""
    with _cli_deadline_scope():
        request = SnapshotDiffRequest(baseline=baseline, current=current)
        result = run_structure_diff(request=request, root=root)
        _emit_structure_diff(result)


@app.command("decision-diff")
def decision_diff(
    baseline: Path = typer.Option(..., "--baseline"),
    current: Path = typer.Option(..., "--current"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Compare two decision surface snapshots and emit a JSON diff."""
    with _cli_deadline_scope():
        request = SnapshotDiffRequest(baseline=baseline, current=current)
        result = run_decision_diff(request=request, root=root)
        _emit_decision_diff(result)


@app.command("structure-reuse")
def structure_reuse(
    snapshot: Path = typer.Option(..., "--snapshot"),
    min_count: int = typer.Option(2, "--min-count"),
    lemma_stubs: Optional[Path] = typer.Option(
        None, "--lemma-stubs", help="Write lemma stubs to file or '-' for stdout."
    ),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Detect repeated subtrees in a structure snapshot."""
    with _cli_deadline_scope():
        result = run_structure_reuse(
            snapshot=snapshot,
            min_count=min_count,
            lemma_stubs=lemma_stubs,
            root=root,
        )
        _emit_structure_reuse(result)


@app.command("refactor-protocol")
def refactor_protocol(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing the refactor request."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write refactor response JSON to this path."
    ),
    rewrite_kind: Literal["protocol_extract", "loop_generator"] = typer.Option(
        "protocol_extract",
        "--rewrite-kind",
        help="Refactor rewrite mode.",
    ),
    protocol_name: Optional[str] = typer.Option(None, "--protocol-name"),
    bundle: Optional[List[str]] = typer.Option(None, "--bundle"),
    field: Optional[List[str]] = typer.Option(
        None,
        "--field",
        help="Field spec in 'name:type' form (repeatable).",
    ),
    target_path: Optional[Path] = typer.Option(None, "--target-path"),
    target_functions: Optional[List[str]] = typer.Option(None, "--target-function"),
    target_loop_lines: Optional[List[int]] = typer.Option(None, "--target-loop-line"),
    compatibility_shim: bool = typer.Option(
        False, "--compat-shim/--no-compat-shim"
    ),
    compatibility_shim_warnings: bool = typer.Option(
        True, "--compat-shim-warnings/--no-compat-shim-warnings"
    ),
    compatibility_shim_overloads: bool = typer.Option(
        True, "--compat-shim-overloads/--no-compat-shim-overloads"
    ),
    ambient_rewrite: bool = typer.Option(False, "--ambient-rewrite/--no-ambient-rewrite"),
    rationale: Optional[str] = typer.Option(None, "--rationale"),
) -> None:
    """Generate protocol refactor edits from a JSON payload (prototype)."""
    with _cli_deadline_scope():
        _run_refactor_protocol(
            input_path=input_path,
            output_path=output_path,
            rewrite_kind=rewrite_kind,
            protocol_name=protocol_name,
            bundle=bundle,
            field=field,
            target_path=target_path,
            target_functions=target_functions,
            target_loop_lines=target_loop_lines,
            compatibility_shim=compatibility_shim,
            compatibility_shim_warnings=compatibility_shim_warnings,
            compatibility_shim_overloads=compatibility_shim_overloads,
            ambient_rewrite=ambient_rewrite,
            rationale=rationale,
        )


_run_refactor_protocol = cast(
    Callable[..., None],
    partial(
        _run_refactor_protocol_impl,
        build_refactor_payload_fn=build_refactor_payload,
        dispatch_command_fn=dispatch_command,
        refactor_command=REFACTOR_COMMAND,
        response_model_validate_fn=RefactorProtocolResponseDTO.model_validate,
        write_text_to_target_fn=_write_text_to_target,
    ),
)
