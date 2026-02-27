from __future__ import annotations
# gabion:decision_protocol_module
# gabion:boundary_normalization_module

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from collections import OrderedDict
from enum import Enum
from typing import Callable, Generator, List, Mapping, MutableMapping, Optional, TypeAlias, cast
import argparse
import inspect
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from click.core import ParameterSource
import typer
from gabion.analysis.timeout_context import (
    check_deadline,
    deadline_loop_iter,
    render_deadline_profile_markdown,
)
from gabion.commands import (
    boundary_order,
    check_contract,
    command_ids,
    progress_contract as progress_timeline,
    transport_policy,
)
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
    CommandRequest,
    run_command,
    run_command_direct,
)
from gabion.plan import (
    ExecutionPlan,
    ExecutionPlanObligations,
    ExecutionPlanPolicyMetadata,
)
from gabion.tooling import (
    delta_advisory as tooling_delta_advisory,
    docflow_delta_emit as tooling_docflow_delta_emit,
    governance_audit as tooling_governance_audit,
    impact_select_tests as tooling_impact_select_tests,
    tool_specs,
    run_dataflow_stage as tooling_run_dataflow_stage,
    ambiguity_contract_policy_check as tooling_ambiguity_contract_policy_check,
    normative_symdiff as tooling_normative_symdiff,
)
from gabion.json_types import JSONObject
from gabion.invariants import never
from gabion.order_contract import sort_once
from gabion.schema import (
    DataflowAuditResponseDTO,
    DecisionDiffResponseDTO,
    RefactorProtocolResponseDTO,
    LspParityGateResponseDTO,
    StructureDiffResponseDTO,
    StructureReuseResponseDTO,
    SynthesisPlanResponseDTO,
)
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
app.add_typer(check_app, name="check")
check_app.add_typer(check_obsolescence_app, name="obsolescence")
check_app.add_typer(check_annotation_drift_app, name="annotation-drift")
check_app.add_typer(check_ambiguity_app, name="ambiguity")
Runner: TypeAlias = Callable[..., JSONObject]
DEFAULT_RUNNER: Runner = run_command

CliRunDataflowRawArgvFn: TypeAlias = Callable[[list[str]], None]
CliRunCheckFn: TypeAlias = Callable[..., JSONObject]
CliRunSppfSyncFn: TypeAlias = Callable[..., int]
CliRunCheckDeltaGatesFn: TypeAlias = Callable[[], int]

_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")

_DEFAULT_TIMEOUT_TICKS = 100
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_CHECK_REPORT_REL_PATH = path_policy.DEFAULT_CHECK_REPORT_REL_PATH
_SPPF_GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
_SPPF_KEYWORD_REF_RE = re.compile(
    r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE
)
_SPPF_PLACEHOLDER_ISSUE_BY_COMMIT: dict[str, str] = {
    "683da24bd121524dc48c218d9771dfbdf181d6f0": "214",
    "61c5d617e7b1d4e734a476adf69bc92c19f35e0f": "214",
}

_LSP_PROGRESS_NOTIFICATION_METHOD = progress_timeline.LSP_PROGRESS_NOTIFICATION_METHOD
_LSP_PROGRESS_TOKEN = progress_timeline.LSP_PROGRESS_TOKEN
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
    )


@app.callback()
def configure_runtime_flags(
    timeout: Optional[str] = typer.Option(
        None,
        "--timeout",
        help="Runtime timeout duration (for example: 750ms, 2s, 1m30s).",
    ),
    carrier: Optional[CliTransportMode] = typer.Option(
        None,
        "--carrier",
        help="Command transport carrier override.",
    ),
    carrier_override_record: Optional[Path] = typer.Option(
        None,
        "--carrier-override-record",
        help="Path to override lifecycle record for direct carrier on governed commands.",
    ),
    removed_lsp_timeout_ticks: Optional[int] = typer.Option(
        None,
        "--lsp-timeout-ticks",
        hidden=True,
    ),
    removed_lsp_timeout_tick_ns: Optional[int] = typer.Option(
        None,
        "--lsp-timeout-tick-ns",
        hidden=True,
    ),
    removed_lsp_timeout_ms: Optional[int] = typer.Option(
        None,
        "--lsp-timeout-ms",
        hidden=True,
    ),
    removed_lsp_timeout_seconds: Optional[float] = typer.Option(
        None,
        "--lsp-timeout-seconds",
        hidden=True,
    ),
    removed_transport: Optional[str] = typer.Option(
        None,
        "--transport",
        hidden=True,
    ),
    removed_direct_run_override_evidence: Optional[str] = typer.Option(
        None,
        "--direct-run-override-evidence",
        hidden=True,
    ),
    removed_override_record_json: Optional[str] = typer.Option(
        None,
        "--override-record-json",
        hidden=True,
    ),
) -> None:
    if (
        removed_lsp_timeout_ticks is not None
        or removed_lsp_timeout_tick_ns is not None
        or removed_lsp_timeout_ms is not None
        or removed_lsp_timeout_seconds is not None
    ):
        raise typer.BadParameter(
            "Removed timeout flags (--lsp-timeout-*). Use --timeout <duration>."
        )
    if (
        removed_transport is not None
        or removed_direct_run_override_evidence is not None
        or removed_override_record_json is not None
    ):
        raise typer.BadParameter(
            "Removed transport flags (--transport/--direct-run-override-evidence/--override-record-json). "
            "Use --carrier and --carrier-override-record."
        )
    policy_runtime.apply_runtime_policy_from_env()
    env_policy.apply_cli_timeout_flag(timeout=timeout)
    carrier_text = None if carrier is None else str(carrier.value)
    carrier_override_record_text = (
        None
        if carrier_override_record is None
        else str(carrier_override_record)
    )
    transport_policy.apply_cli_transport_flags(
        carrier=carrier_text,
        override_record_path=carrier_override_record_text,
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


def _write_lint_sarif(target: str, entries: list[dict[str, object]]) -> None:
    check_deadline()
    rules: dict[str, dict[str, object]] = {}
    rule_counts: dict[str, int] = {}
    results: list[dict[str, object]] = []
    for entry in entries:
        check_deadline()
        code = str(entry.get("code") or "GABION")
        message = str(entry.get("message") or "").strip()
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 1)
        col = int(entry.get("col") or 1)
        rule_counts[code] = int(rule_counts.get(code, 0)) + 1
        rules[code] = {
            "id": code,
            "name": code,
            "shortDescription": {"text": code},
        }
        results.append(
            {
                "ruleId": code,
                "level": "warning",
                "message": {"text": message or code},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": path},
                            "region": {
                                "startLine": line,
                                "startColumn": col,
                            },
                        }
                    }
                ],
            }
        )
    duplicate_codes = sort_once(
        (code for code, count in rule_counts.items() if int(count) > 1),
        source="gabion.cli._emit_lint_sarif.duplicate_codes",
    )
    if duplicate_codes:
        joined = ", ".join(duplicate_codes)
        raise ValueError(f"duplicate SARIF rule code(s): {joined}")
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "gabion", "rules": list(rules.values())}},
                "results": results,
            }
        ],
    }
    payload = json.dumps(sarif, indent=2, sort_keys=False)
    _write_text_to_target(target, payload, ensure_trailing_newline=True)


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


def _render_timeout_progress_markdown(
    *,
    analysis_state: str | None,
    progress: Mapping[str, object],
    deadline_profile: Mapping[str, object] | None = None,
) -> str:
    lines = ["# Timeout Progress", ""]
    if analysis_state:
        lines.append(f"- `analysis_state`: `{analysis_state}`")
    classification = progress.get("classification")
    if isinstance(classification, str):
        lines.append(f"- `classification`: `{classification}`")
    retry_recommended = progress.get("retry_recommended")
    if isinstance(retry_recommended, bool):
        lines.append(f"- `retry_recommended`: `{retry_recommended}`")
    resume_supported = progress.get("resume_supported")
    if isinstance(resume_supported, bool):
        lines.append(f"- `resume_supported`: `{resume_supported}`")
    ticks_consumed = progress.get("ticks_consumed")
    if isinstance(ticks_consumed, int):
        lines.append(f"- `ticks_consumed`: `{ticks_consumed}`")
    tick_limit = progress.get("tick_limit")
    if isinstance(tick_limit, int):
        lines.append(f"- `tick_limit`: `{tick_limit}`")
    ticks_remaining = progress.get("ticks_remaining")
    if isinstance(ticks_remaining, int):
        lines.append(f"- `ticks_remaining`: `{ticks_remaining}`")
    progress_ticks_per_ns = progress.get("ticks_per_ns")
    resolved_ticks_per_ns = (
        progress_ticks_per_ns
        if isinstance(progress_ticks_per_ns, (int, float))
        else (
            deadline_profile.get("ticks_per_ns")
            if isinstance(deadline_profile, Mapping)
            else None
        )
    )
    if isinstance(resolved_ticks_per_ns, (int, float)):
        lines.append(f"- `ticks_per_ns`: `{float(resolved_ticks_per_ns):.9f}`")
    resume = progress.get("resume")
    if isinstance(resume, Mapping):
        token = resume.get("resume_token")
        if isinstance(token, Mapping):
            lines.append("")
            lines.append("## Resume Token")
            lines.append("")
            for key in deadline_loop_iter(
                (
                    "phase",
                    "checkpoint_path",
                    "completed_files",
                    "remaining_files",
                    "total_files",
                    "witness_digest",
                )
            ):
                value = token.get(key)
                if value is None:
                    continue
                lines.append(f"- `{key}`: `{value}`")
    obligations = progress.get("incremental_obligations")
    if isinstance(obligations, list) and obligations:
        lines.append("")
        lines.append("## Incremental Obligations")
        lines.append("")
        for entry in deadline_loop_iter(obligations):
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status", "UNKNOWN") or "UNKNOWN")
            contract = str(entry.get("contract", "") or "")
            kind = str(entry.get("kind", "") or "")
            detail = str(entry.get("detail", "") or "")
            section_id = str(entry.get("section_id", "") or "")
            section_suffix = f" section={section_id}" if section_id else ""
            lines.append(
                f"- `{status}` `{contract}` `{kind}`{section_suffix}: {detail}"
            )
    return "\n".join(lines)


def _build_dataflow_payload_common(
    *,
    options: DataflowPayloadCommonOptions,
) -> JSONObject:
    # dataflow-bundle: filter_bundle
    # dataflow-bundle: deadline_profile
    return check_contract.build_dataflow_payload_common(options=options)


def build_check_payload(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: CheckDeltaOptions,
    exclude: Optional[List[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    lint: bool,
    analysis_tick_limit: int | None = None,
    aux_operation: CheckAuxOperation | None = None,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: Optional[List[Path]] = None,
    aspf_equivalence_against: Optional[List[Path]] = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: Optional[List[Path]] = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: Optional[List[str]] = None,
) -> JSONObject:
    return check_contract.build_check_payload(
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write,
        decision_snapshot=decision_snapshot,
        artifact_flags=artifact_flags,
        delta_options=delta_options,
        exclude=exclude,
        filter_bundle=filter_bundle,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        lint=lint,
        analysis_tick_limit=analysis_tick_limit,
        aux_operation=aux_operation,
        aspf_trace_json=aspf_trace_json,
        aspf_import_trace=aspf_import_trace,
        aspf_equivalence_against=aspf_equivalence_against,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_semantic_surface=aspf_semantic_surface,
    )




def _check_derived_artifacts(
    *,
    report: Path,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    emit_test_obsolescence_delta: bool,
    emit_test_annotation_drift_delta: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    aspf_trace_json: Path | None,
    aspf_opportunities_json: Path | None,
    aspf_state_json: Path | None,
    aspf_delta_jsonl: Path | None,
    aspf_equivalence_enabled: bool,
) -> list[str]:
    derived = [str(report), "artifacts/out/execution_plan.json"]
    if decision_snapshot is not None:
        derived.append(str(decision_snapshot))
    if artifact_flags.emit_test_obsolescence:
        derived.append("artifacts/out/test_obsolescence_report.json")
    if emit_test_obsolescence_state:
        derived.append("artifacts/out/test_obsolescence_state.json")
    if emit_test_obsolescence_delta:
        derived.append("artifacts/out/test_obsolescence_delta.json")
    if artifact_flags.emit_test_evidence_suggestions:
        derived.append("artifacts/out/test_evidence_suggestions.json")
    if artifact_flags.emit_call_clusters:
        derived.append("artifacts/out/call_clusters.json")
    if artifact_flags.emit_call_cluster_consolidation:
        derived.append("artifacts/out/call_cluster_consolidation.json")
    if artifact_flags.emit_test_annotation_drift:
        derived.append("artifacts/out/test_annotation_drift.json")
    if artifact_flags.emit_semantic_coverage_map:
        derived.append("artifacts/out/semantic_coverage_map.json")
    if emit_test_annotation_drift_delta:
        derived.append("artifacts/out/test_annotation_drift_delta.json")
    if emit_ambiguity_delta:
        derived.append("artifacts/out/ambiguity_delta.json")
    if emit_ambiguity_state:
        derived.append("artifacts/out/ambiguity_state.json")
    aspf_enabled = (
        aspf_trace_json is not None
        or aspf_opportunities_json is not None
        or aspf_state_json is not None
        or aspf_equivalence_enabled
    )
    if aspf_enabled:
        derived.append(
            str(aspf_trace_json)
            if aspf_trace_json is not None
            else "artifacts/out/aspf_trace.json"
        )
        derived.append("artifacts/out/aspf_equivalence.json")
        derived.append(
            str(aspf_opportunities_json)
            if aspf_opportunities_json is not None
            else "artifacts/out/aspf_opportunities.json"
        )
        derived.append(
            str(aspf_state_json)
            if aspf_state_json is not None
            else "artifacts/out/aspf_state.json"
        )
        derived.append(
            str(aspf_delta_jsonl)
            if aspf_delta_jsonl is not None
            else "artifacts/out/aspf_delta.jsonl"
        )
    return derived


def build_check_execution_plan_request(
    *,
    payload: JSONObject,
    report: Path,
    decision_snapshot: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    policy: CheckPolicyFlags,
    profile: str,
    artifact_flags: CheckArtifactFlags,
    emit_test_obsolescence_state: bool,
    emit_test_obsolescence_delta: bool,
    emit_test_annotation_drift_delta: bool,
    emit_ambiguity_delta: bool,
    emit_ambiguity_state: bool,
    aspf_trace_json: Path | None = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_equivalence_enabled: bool = False,
) -> ExecutionPlanRequest:
    operations = [DATAFLOW_COMMAND, CHECK_COMMAND]
    obligations = ExecutionPlanObligations(
        preconditions=[
            "input paths resolve under root",
            "analysis timeout budget is configured",
        ],
        postconditions=[
            "exit_code reflects policy gates",
            "execution plan artifact is emitted",
        ],
    )
    baseline_mode = "read"
    if baseline is None:
        baseline_mode = "none"
    elif baseline_write:
        baseline_mode = "write"
    policy_metadata = ExecutionPlanPolicyMetadata(
        deadline={
            "analysis_timeout_ticks": int(payload.get("analysis_timeout_ticks") or 0),
            "analysis_timeout_tick_ns": int(payload.get("analysis_timeout_tick_ns") or 0),
        },
        baseline_mode=baseline_mode,
        docflow_mode="disabled",
    )
    plan = ExecutionPlan(
        requested_operations=operations,
        inputs=dict(payload),
        derived_artifacts=_check_derived_artifacts(
            report=report,
            decision_snapshot=decision_snapshot,
            artifact_flags=artifact_flags,
            emit_test_obsolescence_state=emit_test_obsolescence_state,
            emit_test_obsolescence_delta=emit_test_obsolescence_delta,
            emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
            emit_ambiguity_delta=emit_ambiguity_delta,
            emit_ambiguity_state=emit_ambiguity_state,
            aspf_trace_json=aspf_trace_json,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_equivalence_enabled=aspf_equivalence_enabled,
        ),
        obligations=obligations,
        policy_metadata=policy_metadata,
    )
    plan_payload = plan.as_json_dict()
    plan_payload["policy_metadata"] = dict(plan_payload["policy_metadata"])
    plan_payload["policy_metadata"]["check_profile"] = profile
    plan_payload["policy_metadata"]["fail_on_violations"] = bool(
        policy.fail_on_violations
    )
    plan_payload["policy_metadata"]["fail_on_type_ambiguities"] = bool(
        policy.fail_on_type_ambiguities
    )
    plan_payload["policy_metadata"]["lint"] = bool(policy.lint)
    return ExecutionPlanRequest(
        requested_operations=list(plan_payload["requested_operations"]),
        inputs=dict(plan_payload["inputs"]),
        derived_artifacts=list(plan_payload["derived_artifacts"]),
        obligations=dict(plan_payload["obligations"]),
        policy_metadata=dict(plan_payload["policy_metadata"]),
    )
def parse_dataflow_args_or_exit(
    argv: list[str],
    *,
    parser_fn: Callable[[], argparse.ArgumentParser] | None = None,
) -> argparse.Namespace:
    parser = (parser_fn or dataflow_cli_parser)()
    if any(arg in {"-h", "--help"} for arg in argv):
        parser.print_help()
        raise typer.Exit(code=0)
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        raise typer.Exit(code=int(exc.code))


def build_dataflow_payload(opts: argparse.Namespace) -> JSONObject:
    report_target = _normalize_optional_output_target(opts.report)
    decision_snapshot_target = _normalize_optional_output_target(
        opts.emit_decision_snapshot
    )
    dot_target = _normalize_optional_output_target(opts.dot)
    synthesis_plan_target = _normalize_optional_output_target(opts.synthesis_plan)
    synthesis_protocols_target = _normalize_optional_output_target(
        opts.synthesis_protocols
    )
    refactor_plan_json_target = _normalize_optional_output_target(opts.refactor_plan_json)
    fingerprint_synth_json_target = _normalize_optional_output_target(
        opts.fingerprint_synth_json
    )
    fingerprint_provenance_json_target = _normalize_optional_output_target(
        opts.fingerprint_provenance_json
    )
    fingerprint_deadness_json_target = _normalize_optional_output_target(
        opts.fingerprint_deadness_json
    )
    fingerprint_coherence_json_target = _normalize_optional_output_target(
        opts.fingerprint_coherence_json
    )
    fingerprint_rewrite_plans_json_target = _normalize_optional_output_target(
        opts.fingerprint_rewrite_plans_json
    )
    fingerprint_exception_obligations_json_target = _normalize_optional_output_target(
        opts.fingerprint_exception_obligations_json
    )
    fingerprint_handledness_json_target = _normalize_optional_output_target(
        opts.fingerprint_handledness_json
    )
    aspf_trace_json_target = _normalize_optional_output_target(opts.aspf_trace_json)
    aspf_opportunities_json_target = _normalize_optional_output_target(
        opts.aspf_opportunities_json
    )
    aspf_state_json_target = _normalize_optional_output_target(opts.aspf_state_json)
    aspf_delta_jsonl_target = _normalize_optional_output_target(opts.aspf_delta_jsonl)
    structure_tree_target = _normalize_optional_output_target(opts.emit_structure_tree)
    structure_metrics_target = _normalize_optional_output_target(
        opts.emit_structure_metrics
    )
    aspf_import_trace = (
        check_contract.split_csv_entries(opts.aspf_import_trace)
        if opts.aspf_import_trace
        else []
    )
    aspf_equivalence_against = (
        check_contract.split_csv_entries(opts.aspf_equivalence_against)
        if opts.aspf_equivalence_against
        else []
    )
    aspf_import_state = (
        check_contract.split_csv_entries(opts.aspf_import_state)
        if opts.aspf_import_state
        else []
    )
    aspf_semantic_surface = (
        check_contract.split_csv_entries(opts.aspf_semantic_surface)
        if opts.aspf_semantic_surface
        else []
    )
    payload = _build_dataflow_payload_common(
        options=DataflowPayloadCommonOptions(
            paths=opts.paths,
            root=Path(opts.root),
            config=Path(opts.config) if opts.config is not None else None,
            report=Path(report_target) if report_target else None,
            fail_on_violations=opts.fail_on_violations,
            fail_on_type_ambiguities=opts.fail_on_type_ambiguities,
            baseline=Path(opts.baseline) if opts.baseline else None,
            baseline_write=opts.baseline_write if opts.baseline else None,
            decision_snapshot=Path(decision_snapshot_target)
            if decision_snapshot_target
            else None,
            exclude=opts.exclude,
            filter_bundle=DataflowFilterBundle(
                ignore_params_csv=opts.ignore_params,
                transparent_decorators_csv=opts.transparent_decorators,
            ),
            allow_external=opts.allow_external,
            strictness=opts.strictness,
            lint=bool(opts.lint or opts.lint_jsonl or opts.lint_sarif),
            language=opts.language,
            ingest_profile=opts.ingest_profile,
            aspf_trace_json=Path(aspf_trace_json_target)
            if aspf_trace_json_target
            else None,
            aspf_import_trace=[Path(path) for path in aspf_import_trace],
            aspf_equivalence_against=[Path(path) for path in aspf_equivalence_against],
            aspf_opportunities_json=Path(aspf_opportunities_json_target)
            if aspf_opportunities_json_target
            else None,
            aspf_state_json=Path(aspf_state_json_target)
            if aspf_state_json_target
            else None,
            aspf_import_state=[Path(path) for path in aspf_import_state],
            aspf_delta_jsonl=Path(aspf_delta_jsonl_target)
            if aspf_delta_jsonl_target
            else None,
            aspf_semantic_surface=list(aspf_semantic_surface),
        )
    )
    payload.update(
        {
        "dot": dot_target,
        "no_recursive": opts.no_recursive,
        "max_components": opts.max_components,
        "type_audit": opts.type_audit,
        "type_audit_report": opts.type_audit_report,
        "type_audit_max": opts.type_audit_max,
        "synthesis_plan": synthesis_plan_target,
        "synthesis_report": opts.synthesis_report,
        "synthesis_max_tier": opts.synthesis_max_tier,
        "synthesis_min_bundle_size": opts.synthesis_min_bundle_size,
        "synthesis_allow_singletons": opts.synthesis_allow_singletons,
        "synthesis_protocols": synthesis_protocols_target,
        "synthesis_protocols_kind": opts.synthesis_protocols_kind,
        "refactor_plan": opts.refactor_plan,
        "refactor_plan_json": refactor_plan_json_target,
        "fingerprint_synth_json": fingerprint_synth_json_target,
        "fingerprint_provenance_json": fingerprint_provenance_json_target,
        "fingerprint_deadness_json": fingerprint_deadness_json_target,
        "fingerprint_coherence_json": fingerprint_coherence_json_target,
        "fingerprint_rewrite_plans_json": fingerprint_rewrite_plans_json_target,
        "fingerprint_exception_obligations_json": (
            fingerprint_exception_obligations_json_target
        ),
        "fingerprint_handledness_json": fingerprint_handledness_json_target,
        "synthesis_merge_overlap": opts.synthesis_merge_overlap,
        "structure_tree": structure_tree_target,
        "structure_metrics": structure_metrics_target,
        "proof_mode": getattr(opts, "proof_mode", None),
        "order_policy": getattr(opts, "order_policy", None),
        "order_telemetry": getattr(opts, "order_telemetry", None),
        "order_enforce_canonical_allowlist": getattr(opts, "order_enforce_canonical_allowlist", None),
        "order_deadline_probe": getattr(opts, "order_deadline_probe", None),
        "derivation_cache_max_entries": getattr(opts, "derivation_cache_max_entries", None),
        "projection_registry_gas_limit": getattr(opts, "projection_registry_gas_limit", None),
        }
    )
    return payload


def build_refactor_payload(
    *,
    input_payload: Optional[JSONObject] = None,
    protocol_name: Optional[str],
    bundle: Optional[List[str]],
    field: Optional[List[str]],
    target_path: Optional[Path],
    target_functions: Optional[List[str]],
    compatibility_shim: bool,
    compatibility_shim_warnings: bool,
    compatibility_shim_overloads: bool,
    ambient_rewrite: bool,
    rationale: Optional[str],
) -> JSONObject:
    check_deadline()
    if input_payload is not None:
        return input_payload
    if protocol_name is None or target_path is None:
        raise typer.BadParameter(
            "Provide --protocol-name and --target-path or use --input."
        )
    field_specs: list[dict[str, str | None]] = []
    for spec in field or []:
        check_deadline()
        name, _, hint = spec.partition(":")
        name = name.strip()
        if not name:
            continue
        type_hint = hint.strip() or None
        field_specs.append({"name": name, "type_hint": type_hint})
    if not bundle and field_specs:
        bundle = [spec["name"] for spec in field_specs]
    compatibility_shim_payload: bool | dict[str, bool]
    if compatibility_shim:
        compatibility_shim_payload = {
            "enabled": True,
            "emit_deprecation_warning": compatibility_shim_warnings,
            "emit_overload_stubs": compatibility_shim_overloads,
        }
    else:
        compatibility_shim_payload = False
    return {
        "protocol_name": protocol_name,
        "bundle": bundle or [],
        "fields": field_specs,
        "target_path": str(target_path),
        "target_functions": target_functions or [],
        "compatibility_shim": compatibility_shim_payload,
        "ambient_rewrite": ambient_rewrite,
        "rationale": rationale,
    }


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
    def _ordered_result(
        value: Mapping[str, object],
    ) -> JSONObject:
        return boundary_order.normalize_boundary_mapping_once(
            value,
            source=f"cli.dispatch_command.{command}.result",
        )

    ticks, tick_ns = _cli_timeout_ticks()
    payload = boundary_order.normalize_boundary_mapping_once(
        payload,
        source=f"cli.dispatch_command.{command}.payload_in",
    )
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        payload = boundary_order.apply_boundary_updates_once(
            payload,
            {
                "analysis_timeout_ticks": int(ticks),
                "analysis_timeout_tick_ns": int(tick_ns),
            },
            source=f"cli.dispatch_command.{command}.payload_timeout_defaults",
        )
    if execution_plan_request is not None:
        execution_plan_payload = execution_plan_request.to_payload()
        execution_plan_inputs = execution_plan_payload.get("inputs")
        if isinstance(execution_plan_inputs, Mapping):
            merged_inputs = boundary_order.apply_boundary_updates_once(
                execution_plan_inputs,
                payload,
                source=f"cli.dispatch_command.{command}.execution_plan_inputs",
            )
            execution_plan_payload["inputs"] = merged_inputs
        deadline_metadata = execution_plan_payload.get("policy_metadata")
        if isinstance(deadline_metadata, Mapping):
            policy_metadata = dict(deadline_metadata)
            deadline = policy_metadata.get("deadline")
            deadline_payload = dict(deadline) if isinstance(deadline, Mapping) else {}
            deadline_payload = boundary_order.apply_boundary_updates_once(
                deadline_payload,
                {
                    "analysis_timeout_ticks": int(
                        payload.get("analysis_timeout_ticks") or 0
                    ),
                    "analysis_timeout_tick_ns": int(
                        payload.get("analysis_timeout_tick_ns") or 0
                    ),
                },
                source=f"cli.dispatch_command.{command}.execution_plan_deadline",
            )
            policy_metadata["deadline"] = deadline_payload
            execution_plan_payload["policy_metadata"] = policy_metadata
        execution_plan_payload = boundary_order.normalize_boundary_mapping_once(
            execution_plan_payload,
            source=f"cli.dispatch_command.{command}.execution_plan_payload",
        )
        payload = boundary_order.apply_boundary_updates_once(
            payload,
            {"execution_plan_request": execution_plan_payload},
            source=f"cli.dispatch_command.{command}.payload_execution_plan",
        )
    payload = boundary_order.enforce_boundary_mapping_ordered(
        payload,
        source=f"cli.dispatch_command.{command}.payload_out",
    )
    request = CommandRequest(command, [payload])
    transport = transport_policy.resolve_command_transport(command=command, runner=runner)
    resolved = transport.runner
    if resolved is run_command:
        factory = process_factory or subprocess.Popen
        raw = resolved(
            request,
            root=root,
            timeout_ticks=ticks,
            timeout_tick_ns=tick_ns,
            process_factory=factory,
            notification_callback=notification_callback,
        )
    elif resolved is run_command_direct:
        raw = resolved(
            request,
            root=root,
            notification_callback=notification_callback,
        )
    else:
        if notification_callback is not None:
            try:
                params = inspect.signature(resolved).parameters
            except (TypeError, ValueError):
                params = {}
            if "notification_callback" in params or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            ):
                raw = resolved(
                    request,
                    root=root,
                    notification_callback=notification_callback,
                )
            else:
                raw = resolved(request, root=root)
        else:
            raw = resolved(request, root=root)
    if not isinstance(raw, Mapping):
        never(
            "command returned non-mapping payload",
            command=command,
            result_type=type(raw).__name__,
        )
    return _ordered_result(raw)


def run_check(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    policy: CheckPolicyFlags,
    root: Path,
    config: Optional[Path],
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: CheckDeltaOptions,
    exclude: Optional[List[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    analysis_tick_limit: int | None = None,
    aux_operation: CheckAuxOperation | None = None,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: Optional[List[Path]] = None,
    aspf_equivalence_against: Optional[List[Path]] = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: Optional[List[Path]] = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: Optional[List[str]] = None,
    runner: Runner = run_command,
    notification_callback: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    if filter_bundle is None:
        filter_bundle = DataflowFilterBundle(None, None)
    # dataflow-bundle: filter_bundle
    resolved_report = _resolve_check_report_path(report, root=root)
    resolved_report.parent.mkdir(parents=True, exist_ok=True)
    payload = build_check_payload(
        paths=paths,
        report=resolved_report,
        fail_on_violations=policy.fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write if baseline is not None else False,
        decision_snapshot=decision_snapshot,
        artifact_flags=artifact_flags,
        delta_options=delta_options,
        exclude=exclude,
        filter_bundle=filter_bundle,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=policy.fail_on_type_ambiguities,
        lint=policy.lint,
        analysis_tick_limit=analysis_tick_limit,
        aux_operation=aux_operation,
        aspf_trace_json=aspf_trace_json,
        aspf_import_trace=aspf_import_trace,
        aspf_equivalence_against=aspf_equivalence_against,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_semantic_surface=aspf_semantic_surface,
    )
    execution_plan_request = build_check_execution_plan_request(
        payload=payload,
        report=resolved_report,
        decision_snapshot=decision_snapshot,
        baseline=baseline,
        baseline_write=baseline_write,
        policy=policy,
        profile="strict",
        artifact_flags=artifact_flags,
        emit_test_obsolescence_state=delta_options.emit_test_obsolescence_state,
        emit_test_obsolescence_delta=delta_options.emit_test_obsolescence_delta,
        emit_test_annotation_drift_delta=delta_options.emit_test_annotation_drift_delta,
        emit_ambiguity_delta=delta_options.emit_ambiguity_delta,
        emit_ambiguity_state=delta_options.emit_ambiguity_state,
        aspf_trace_json=aspf_trace_json,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_equivalence_enabled=bool(
            aspf_trace_json
            or aspf_import_trace
            or aspf_equivalence_against
            or aspf_opportunities_json
            or aspf_state_json
            or aspf_import_state
            or aspf_delta_jsonl
            or aspf_semantic_surface
        ),
    )
    return dispatch_command(
        command=DATAFLOW_COMMAND,
        payload=payload,
        root=root,
        runner=runner,
        execution_plan_request=execution_plan_request,
        notification_callback=notification_callback,
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
    with _cli_deadline_scope():
        normalized_result = DataflowAuditResponseDTO.model_validate(
            {
                "exit_code": int(result.get("exit_code", 0) or 0),
                "timeout": bool(result.get("timeout", False)),
                "analysis_state": result.get("analysis_state"),
                "errors": result.get("errors") or [],
                "lint_lines": result.get("lint_lines") or [],
                "lint_entries": result.get("lint_entries") or [],
                "payload": result,
            }
        ).model_dump()
        lint_lines = normalized_result.get("lint_lines", []) or []
        lint_entries_raw = normalized_result.get("lint_entries")
        lint_entries = lint_entries_raw if isinstance(lint_entries_raw, list) else None
        _emit_lint_outputs(
            lint_lines,
            lint=opts.lint,
            lint_jsonl=opts.lint_jsonl,
            lint_sarif=opts.lint_sarif,
            lint_entries=lint_entries,
        )
        if opts.type_audit:
            suggestions = result.get("type_suggestions", [])
            ambiguities = result.get("type_ambiguities", [])
            if suggestions:
                typer.echo("Type tightening candidates:")
                for line in suggestions[: opts.type_audit_max]:
                    check_deadline()
                    typer.echo(f"- {line}")
            if ambiguities:
                typer.echo("Type ambiguities (conflicting downstream expectations):")
                for line in ambiguities[: opts.type_audit_max]:
                    check_deadline()
                    typer.echo(f"- {line}")
        if _is_stdout_target(opts.dot) and "dot" in result:
            _write_text_to_target(_STDOUT_PATH, str(result["dot"]), ensure_trailing_newline=True)
        if _is_stdout_target(opts.synthesis_plan) and "synthesis_plan" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["synthesis_plan"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.synthesis_protocols) and "synthesis_protocols" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                str(result["synthesis_protocols"]),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.refactor_plan_json) and "refactor_plan" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["refactor_plan"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_synth_json)
            and "fingerprint_synth_registry" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_synth_registry"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_provenance_json)
            and "fingerprint_provenance" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_provenance"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.fingerprint_deadness_json) and "fingerprint_deadness" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_deadness"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.fingerprint_coherence_json) and "fingerprint_coherence" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_coherence"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_rewrite_plans_json)
            and "fingerprint_rewrite_plans" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_rewrite_plans"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_exception_obligations_json)
            and "fingerprint_exception_obligations" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(
                    result["fingerprint_exception_obligations"],
                    indent=2,
                    sort_keys=False,
                ),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_handledness_json)
            and "fingerprint_handledness" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_handledness"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        stdout_json_targets = (
            (opts.emit_structure_tree, "structure_tree"),
            (opts.emit_structure_metrics, "structure_metrics"),
            (opts.emit_decision_snapshot, "decision_snapshot"),
            (opts.aspf_trace_json, "aspf_trace"),
            (opts.aspf_trace_json, "aspf_equivalence"),
            (opts.aspf_opportunities_json, "aspf_opportunities"),
            (opts.aspf_state_json, "aspf_state"),
        )
        for output_target, result_key in stdout_json_targets:
            if result_key in result and _is_stdout_target(output_target):
                _emit_result_json_to_stdout(payload=result[result_key])


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


def _check_raw_profile_args(
    *,
    ctx: typer.Context,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    decision_snapshot: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
) -> list[str]:
    resolved_filter_bundle = filter_bundle or DataflowFilterBundle(None, None)
    argv = [str(path) for path in (paths or [])]
    if _param_is_command_line(ctx, "root"):
        argv.extend(["--root", str(root)])
    if _param_is_command_line(ctx, "config") and config is not None:
        argv.extend(["--config", str(config)])
    if _param_is_command_line(ctx, "report") and report is not None:
        argv.extend(["--report", str(report)])
    if _param_is_command_line(ctx, "decision_snapshot") and decision_snapshot is not None:
        argv.extend(["--emit-decision-snapshot", str(decision_snapshot)])
    if _param_is_command_line(ctx, "baseline") and baseline is not None:
        argv.extend(["--baseline", str(baseline)])
    if _param_is_command_line(ctx, "baseline_write") and baseline_write:
        argv.append("--baseline-write")
    if _param_is_command_line(ctx, "exclude"):
        for entry in deadline_loop_iter(exclude or []):
            argv.extend(["--exclude", entry])
    if (
        _param_is_command_line(ctx, "ignore_params_csv")
        and resolved_filter_bundle.ignore_params_csv is not None
    ):
        argv.extend(["--ignore-params", resolved_filter_bundle.ignore_params_csv])
    if (
        _param_is_command_line(ctx, "transparent_decorators_csv")
        and resolved_filter_bundle.transparent_decorators_csv is not None
    ):
        argv.extend(["--transparent-decorators", resolved_filter_bundle.transparent_decorators_csv])
    if _param_is_command_line(ctx, "allow_external") and allow_external is not None:
        argv.append("--allow-external" if allow_external else "--no-allow-external")
    if _param_is_command_line(ctx, "strictness") and strictness is not None:
        argv.extend(["--strictness", strictness])
    if _param_is_command_line(ctx, "fail_on_violations") and fail_on_violations:
        argv.append("--fail-on-violations")
    if _param_is_command_line(ctx, "fail_on_type_ambiguities") and fail_on_type_ambiguities:
        argv.append("--fail-on-type-ambiguities")
    if _param_is_command_line(ctx, "lint") and lint:
        argv.append("--lint")
    if _param_is_command_line(ctx, "lint_jsonl") and lint_jsonl is not None:
        argv.extend(["--lint-jsonl", str(lint_jsonl)])
    if _param_is_command_line(ctx, "lint_sarif") and lint_sarif is not None:
        argv.extend(["--lint-sarif", str(lint_sarif)])
    return argv


def _run_check_raw_profile(
    *,
    ctx: typer.Context,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    decision_snapshot: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    lint: bool,
    lint_jsonl: Optional[Path],
    lint_sarif: Optional[Path],
    run_dataflow_raw_argv_fn: Callable[[list[str]], None] | None = None,
) -> None:
    resolved_filter_bundle = filter_bundle or DataflowFilterBundle(None, None)
    unsupported = _raw_profile_unsupported_flags(ctx)
    if unsupported:
        rendered = ", ".join(unsupported)
        raise typer.BadParameter(
            f"--profile raw does not support check-only options: {rendered}"
        )
    raw_args = _check_raw_profile_args(
        ctx=ctx,
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        decision_snapshot=decision_snapshot,
        baseline=baseline,
        baseline_write=baseline_write,
        exclude=exclude,
        filter_bundle=filter_bundle,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
        lint=lint,
        lint_jsonl=lint_jsonl,
        lint_sarif=lint_sarif,
    )
    resolved_run = run_dataflow_raw_argv_fn or _run_dataflow_raw_argv
    resolved_run(raw_args + list(ctx.args))


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


def _run_check_command(
    *,
    ctx: typer.Context,
    paths: list[Path] | None,
    report: Path | None,
    root: Path,
    config: Path | None,
    baseline: Path | None,
    baseline_write: bool,
    decision_snapshot: Path | None,
    artifact_flags: CheckArtifactFlags,
    delta_options: CheckDeltaOptions,
    exclude: list[str] | None,
    filter_bundle: DataflowFilterBundle | None,
    allow_external: bool | None,
    strictness: str | None,
    analysis_budget_checks: int | None,
    gate: CheckGateMode,
    lint_mode: CheckLintMode,
    lint_jsonl_out: Path | None,
    lint_sarif_out: Path | None,
    aspf_trace_json: Path | None,
    aspf_import_trace: list[Path] | None,
    aspf_equivalence_against: list[Path] | None,
    aspf_opportunities_json: Path | None,
    aspf_state_json: Path | None,
    aspf_import_state: list[Path] | None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: list[str] | None = None,
    aux_operation: CheckAuxOperation | None = None,
) -> None:
    fail_on_violations, fail_on_type_ambiguities = _check_gate_policy(gate)
    lint_enabled, lint_line = _check_lint_mode(
        lint_mode=lint_mode,
        lint_jsonl_out=lint_jsonl_out,
        lint_sarif_out=lint_sarif_out,
    )
    deps = _context_cli_deps(ctx)
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
            if not timeline_header_emitted
            and isinstance(header_value, str)
            and header_value
            else None
        )
        _emit_phase_timeline_progress(header=header, row=row)
        if header is not None:
            timeline_header_emitted = True

    result = _run_with_timeout_retries(
        run_once=lambda: deps.run_check_fn(
            paths=paths,
            report=report,
            policy=CheckPolicyFlags(
                fail_on_violations=fail_on_violations,
                fail_on_type_ambiguities=fail_on_type_ambiguities,
                lint=lint_enabled,
            ),
            root=root,
            config=config,
            baseline=baseline,
            baseline_write=baseline_write,
            decision_snapshot=decision_snapshot,
            artifact_flags=artifact_flags,
            delta_options=delta_options,
            exclude=exclude,
            filter_bundle=filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            analysis_tick_limit=analysis_budget_checks,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_semantic_surface=aspf_semantic_surface,
            aux_operation=aux_operation,
            notification_callback=_on_notification,
        ),
        root=Path(root),
    )
    with _cli_deadline_scope():
        lint_lines = result.get("lint_lines", []) or []
        _emit_lint_outputs(
            lint_lines,
            lint=lint_line,
            lint_jsonl=lint_jsonl_out,
            lint_sarif=lint_sarif_out,
        )
    _emit_analysis_resume_summary(result)
    _emit_nonzero_exit_causes(result)
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@check_app.command("delta-bundle")
def check_delta_bundle(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    report: Optional[Path] = typer.Option(None, "--report"),
    strictness: CheckStrictnessMode = typer.Option(
        CheckStrictnessMode.high, "--strictness"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    decision_snapshot: Optional[Path] = typer.Option(
        None,
        "--decision-snapshot",
    ),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    aspf_trace_json: Optional[Path] = typer.Option(
        None,
        "--aspf-trace-json",
    ),
    aspf_import_trace: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-trace",
    ),
    aspf_equivalence_against: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-equivalence-against",
    ),
    aspf_opportunities_json: Optional[Path] = typer.Option(
        None,
        "--aspf-opportunities-json",
    ),
    aspf_state_json: Optional[Path] = typer.Option(
        None,
        "--aspf-state-json",
    ),
    aspf_delta_jsonl: Optional[Path] = typer.Option(
        None,
        "--aspf-delta-jsonl",
    ),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
    aspf_semantic_surface: Optional[List[str]] = typer.Option(
        None,
        "--aspf-semantic-surface",
    ),
) -> None:
    _run_check_command(
        ctx=ctx,
        paths=paths,
        report=report,
        root=root,
        config=config,
        baseline=None,
        baseline_write=False,
        decision_snapshot=decision_snapshot,
        artifact_flags=check_contract.delta_bundle_artifact_flags(),
        delta_options=check_contract.delta_bundle_delta_options(),
        exclude=None,
        filter_bundle=DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=allow_external,
        strictness=str(strictness.value),
        analysis_budget_checks=analysis_budget_checks,
        gate=CheckGateMode.none,
        lint_mode=CheckLintMode.none,
        lint_jsonl_out=None,
        lint_sarif_out=None,
        aspf_trace_json=aspf_trace_json,
        aspf_import_trace=aspf_import_trace,
        aspf_equivalence_against=aspf_equivalence_against,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_import_state=aspf_import_state,
        aspf_semantic_surface=aspf_semantic_surface,
    )


@check_app.command("delta-gates")
def check_delta_gates(ctx: typer.Context) -> None:
    deps = _context_cli_deps(ctx)
    raise typer.Exit(code=deps.run_check_delta_gates_fn())


def _run_check_aux_operation(
    *,
    ctx: typer.Context,
    domain: str,
    action: str,
    paths: list[Path] | None,
    root: Path,
    config: Path | None,
    strictness: CheckStrictnessMode,
    allow_external: bool | None,
    baseline: Path | None,
    state_in: Path | None,
    out_json: Path | None,
    out_md: Path | None,
    report: Path | None,
    decision_snapshot: Path | None,
    analysis_budget_checks: int | None,
    aspf_state_json: Path | None,
    aspf_import_state: list[Path] | None,
    aspf_delta_jsonl: Path | None = None,
) -> None:
    aux_operation = CheckAuxOperation(
        domain=domain,
        action=action,
        baseline_path=baseline,
        state_in_path=state_in,
        out_json=out_json,
        out_md=out_md,
    )
    _run_check_command(
        ctx=ctx,
        paths=paths,
        report=report,
        root=root,
        config=config,
        baseline=None,
        baseline_write=False,
        decision_snapshot=decision_snapshot,
        artifact_flags=_default_check_artifact_flags(),
        delta_options=_default_check_delta_options(),
        exclude=None,
        filter_bundle=DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=allow_external,
        strictness=str(strictness.value),
        analysis_budget_checks=analysis_budget_checks,
        gate=CheckGateMode.none,
        lint_mode=CheckLintMode.none,
        lint_jsonl_out=None,
        lint_sarif_out=None,
        aspf_trace_json=None,
        aspf_import_trace=None,
        aspf_equivalence_against=None,
        aspf_opportunities_json=None,
        aspf_state_json=aspf_state_json,
        aspf_import_state=aspf_import_state,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_semantic_surface=None,
        aux_operation=aux_operation,
    )


@check_app.callback()
def check_group(
    ctx: typer.Context,
    removed_profile: Optional[str] = typer.Option(None, "--profile", hidden=True),
    removed_emit_test_obsolescence: bool = typer.Option(
        False,
        "--emit-test-obsolescence",
        hidden=True,
    ),
    removed_emit_test_obsolescence_state: bool = typer.Option(
        False,
        "--emit-test-obsolescence-state",
        hidden=True,
    ),
    removed_emit_test_obsolescence_delta: bool = typer.Option(
        False,
        "--emit-test-obsolescence-delta",
        hidden=True,
    ),
    removed_test_obsolescence_state: Optional[Path] = typer.Option(
        None,
        "--test-obsolescence-state",
        hidden=True,
    ),
    removed_emit_test_annotation_drift: bool = typer.Option(
        False,
        "--emit-test-annotation-drift",
        hidden=True,
    ),
    removed_emit_test_annotation_drift_delta: bool = typer.Option(
        False,
        "--emit-test-annotation-drift-delta",
        hidden=True,
    ),
    removed_write_test_annotation_drift_baseline: bool = typer.Option(
        False,
        "--write-test-annotation-drift-baseline",
        hidden=True,
    ),
    removed_test_annotation_drift_state: Optional[Path] = typer.Option(
        None,
        "--test-annotation-drift-state",
        hidden=True,
    ),
    removed_emit_ambiguity_delta: bool = typer.Option(
        False,
        "--emit-ambiguity-delta",
        hidden=True,
    ),
    removed_emit_ambiguity_state: bool = typer.Option(
        False,
        "--emit-ambiguity-state",
        hidden=True,
    ),
    removed_ambiguity_state: Optional[Path] = typer.Option(
        None,
        "--ambiguity-state",
        hidden=True,
    ),
    removed_write_ambiguity_baseline: bool = typer.Option(
        False,
        "--write-ambiguity-baseline",
        hidden=True,
    ),
    removed_write_test_obsolescence_baseline: bool = typer.Option(
        False,
        "--write-test-obsolescence-baseline",
        hidden=True,
    ),
) -> None:
    if removed_profile is not None:
        raise typer.BadParameter(
            "Removed --profile flag. Use `gabion check run` or `gabion check raw -- ...`."
        )
    if (
        removed_emit_test_obsolescence
        or removed_emit_test_obsolescence_state
        or removed_emit_test_obsolescence_delta
        or removed_test_obsolescence_state is not None
        or removed_emit_test_annotation_drift
        or removed_emit_test_annotation_drift_delta
        or removed_write_test_annotation_drift_baseline
        or removed_test_annotation_drift_state is not None
        or removed_emit_ambiguity_delta
        or removed_emit_ambiguity_state
        or removed_ambiguity_state is not None
        or removed_write_ambiguity_baseline
        or removed_write_test_obsolescence_baseline
    ):
        raise typer.BadParameter(
            "Removed legacy check modality flags. Use `gabion check obsolescence|annotation-drift|ambiguity` subcommands."
        )
    _check_help_or_exit(ctx)


@check_obsolescence_app.callback()
def check_obsolescence_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_annotation_drift_app.callback()
def check_annotation_drift_group(ctx: typer.Context) -> None:
    _check_help_or_exit(ctx)


@check_ambiguity_app.callback()
def check_ambiguity_group(ctx: typer.Context) -> None:
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


@check_app.command("run")
def check_run(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    report: Optional[Path] = typer.Option(None, "--report"),
    strictness: CheckStrictnessMode = typer.Option(CheckStrictnessMode.high, "--strictness"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    baseline: Optional[Path] = typer.Option(None, "--baseline"),
    baseline_mode: CheckBaselineMode = typer.Option(
        CheckBaselineMode.off,
        "--baseline-mode",
    ),
    gate: CheckGateMode = typer.Option(CheckGateMode.all, "--gate"),
    analysis_budget_checks: Optional[int] = typer.Option(
        None,
        "--analysis-budget-checks",
        min=1,
    ),
    removed_analysis_tick_limit: Optional[int] = typer.Option(
        None,
        "--analysis-tick-limit",
        hidden=True,
    ),
    decision_snapshot: Optional[Path] = typer.Option(
        None,
        "--decision-snapshot",
    ),
    lint: CheckLintMode = typer.Option(CheckLintMode.none, "--lint"),
    lint_jsonl_out: Optional[Path] = typer.Option(
        None,
        "--lint-jsonl-out",
    ),
    lint_sarif_out: Optional[Path] = typer.Option(
        None,
        "--lint-sarif-out",
    ),
    aspf_trace_json: Optional[Path] = typer.Option(
        None,
        "--aspf-trace-json",
    ),
    aspf_import_trace: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-trace",
    ),
    aspf_equivalence_against: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-equivalence-against",
    ),
    aspf_opportunities_json: Optional[Path] = typer.Option(
        None,
        "--aspf-opportunities-json",
    ),
    aspf_state_json: Optional[Path] = typer.Option(
        None,
        "--aspf-state-json",
    ),
    aspf_delta_jsonl: Optional[Path] = typer.Option(
        None,
        "--aspf-delta-jsonl",
    ),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
    aspf_semantic_surface: Optional[List[str]] = typer.Option(
        None,
        "--aspf-semantic-surface",
    ),
) -> None:
    if removed_analysis_tick_limit is not None:
        raise typer.BadParameter(
            "Removed --analysis-tick-limit. Use --analysis-budget-checks."
        )
    if baseline_mode in {CheckBaselineMode.enforce, CheckBaselineMode.write} and baseline is None:
        raise typer.BadParameter(
            "--baseline is required when --baseline-mode is enforce or write."
        )
    if baseline_mode is CheckBaselineMode.off and baseline is not None:
        raise typer.BadParameter(
            "--baseline is only valid when --baseline-mode is enforce or write."
        )
    baseline_path = baseline if baseline_mode is not CheckBaselineMode.off else None
    baseline_write = baseline_mode is CheckBaselineMode.write
    _run_check_command(
        ctx=ctx,
        paths=paths,
        report=report,
        root=root,
        config=config,
        baseline=baseline_path,
        baseline_write=baseline_write,
        decision_snapshot=decision_snapshot,
        artifact_flags=_default_check_artifact_flags(),
        delta_options=_default_check_delta_options(),
        exclude=None,
        filter_bundle=DataflowFilterBundle(
            ignore_params_csv=None,
            transparent_decorators_csv=None,
        ),
        allow_external=allow_external,
        strictness=str(strictness.value),
        analysis_budget_checks=analysis_budget_checks,
        gate=gate,
        lint_mode=lint,
        lint_jsonl_out=lint_jsonl_out,
        lint_sarif_out=lint_sarif_out,
        aspf_trace_json=aspf_trace_json,
        aspf_import_trace=aspf_import_trace,
        aspf_equivalence_against=aspf_equivalence_against,
        aspf_opportunities_json=aspf_opportunities_json,
        aspf_state_json=aspf_state_json,
        aspf_delta_jsonl=aspf_delta_jsonl,
        aspf_import_state=aspf_import_state,
        aspf_semantic_surface=aspf_semantic_surface,
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


def dataflow_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dataflow grammar audit in raw profile mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline", default=None, help="Baseline file for violations.")
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write current violations to baseline file.",
    )
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--ignore-params", default=None)
    parser.add_argument(
        "--transparent-decorators",
        default=None,
        help="Comma-separated decorator names treated as transparent.",
    )
    parser.add_argument(
        "--allow-external",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--strictness", choices=["high", "low"], default=None)
    parser.add_argument(
        "--language",
        default=None,
        help="Explicit analysis language adapter (for example: python).",
    )
    parser.add_argument(
        "--ingest-profile",
        default=None,
        help="Optional ingest profile used by the selected language adapter.",
    )
    parser.add_argument(
        "--aspf-trace-json",
        default=None,
        help="Write ASPF execution trace JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--aspf-import-trace",
        action="append",
        default=None,
        help="Import one or more prior ASPF trace JSON artifacts.",
    )
    parser.add_argument(
        "--aspf-equivalence-against",
        action="append",
        default=None,
        help="One or more ASPF trace JSON artifacts used as equivalence baseline.",
    )
    parser.add_argument(
        "--aspf-opportunities-json",
        default=None,
        help="Write ASPF simplification/fungibility opportunities JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--aspf-state-json",
        default=None,
        help="Write ASPF serialized state JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--aspf-delta-jsonl",
        default=None,
        help="Write ASPF mutation delta ledger JSONL to file or '-' for stdout.",
    )
    parser.add_argument(
        "--aspf-import-state",
        action="append",
        default=None,
        help="Import one or more prior ASPF serialized state JSON artifacts.",
    )
    parser.add_argument(
        "--aspf-semantic-surface",
        action="append",
        default=None,
        help="Semantic surface keys to project into ASPF representatives.",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument(
        "--emit-structure-tree",
        default=None,
        help="Write canonical structure snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-structure-metrics",
        default=None,
        help="Write structure metrics JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-synth-json",
        default=None,
        help="Write fingerprint synth registry JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-provenance-json",
        default=None,
        help="Write fingerprint provenance JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-deadness-json",
        default=None,
        help="Write fingerprint deadness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-coherence-json",
        default=None,
        help="Write fingerprint coherence JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-rewrite-plans-json",
        default=None,
        help="Write fingerprint rewrite plans JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-exception-obligations-json",
        default=None,
        help="Write fingerprint exception obligations JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--fingerprint-handledness-json",
        default=None,
        help="Write fingerprint handledness JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--emit-decision-snapshot",
        default=None,
        help="Write decision surface snapshot JSON to file or '-' for stdout.",
    )
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Emit lint-style lines (path:line:col: CODE message).",
    )
    parser.add_argument(
        "--lint-jsonl",
        default=None,
        help="Write lint JSONL to file or '-' for stdout.",
    )
    parser.add_argument(
        "--lint-sarif",
        default=None,
        help="Write lint SARIF to file or '-' for stdout.",
    )
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    parser.add_argument(
        "--type-audit-max",
        type=int,
        default=50,
        help="Max type-tightening entries to print.",
    )
    parser.add_argument(
        "--type-audit-report",
        action="store_true",
        help="Include type-flow audit summary in the markdown report.",
    )
    parser.add_argument(
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-protocols-kind",
        choices=["dataclass", "protocol", "contextvar"],
        default="dataclass",
        help="Emit dataclass, typing.Protocol, or ContextVar stubs (default: dataclass).",
    )
    parser.add_argument(
        "--synthesis-max-tier",
        type=int,
        default=2,
        help="Max tier to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-min-bundle-size",
        type=int,
        default=2,
        help="Min bundle size to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-allow-singletons",
        action="store_true",
        help="Allow single-field bundles in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    parser.add_argument(
        "--refactor-plan",
        action="store_true",
        help="Include refactoring plan summary in the markdown report.",
    )
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
    )
    return parser


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


def _restore_aspf_state_from_github_artifacts(
    *,
    token: str,
    repo: str,
    output_dir: Path,
    ref_name: str = "",
    current_run_id: str = "",
    artifact_name: str = "dataflow-report",
    state_name: str = "aspf_state_ci.json",
    per_page: int = 100,
    urlopen_fn: Callable[..., object] = urllib.request.urlopen,
    no_redirect_open_fn: Callable[..., object] | None = None,
    follow_redirect_open_fn: Callable[..., object] | None = None,
) -> int:
    token = token.strip()
    repo = repo.strip()
    ref_name = ref_name.strip()
    current_run_id = current_run_id.strip()
    artifact_name = artifact_name.strip() or "dataflow-report"
    state_name = state_name.strip() or "aspf_state_ci.json"
    if not token or not repo:
        typer.echo("GitHub token/repository unavailable; skipping ASPF state restore.")
        return 0

    api_url = (
        f"https://api.github.com/repos/{repo}/actions/artifacts"
        f"?name={artifact_name}&per_page={max(1, int(per_page))}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urlopen_fn(req, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        typer.echo(f"Unable to query prior artifacts ({exc}); skipping ASPF state restore.")
        return 0

    artifacts = payload.get("artifacts", []) if isinstance(payload, dict) else []

    def _artifact_is_candidate(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        download_url = str(item.get("archive_download_url", "") or "")
        if item.get("expired", True) or not download_url:
            return False
        workflow_run = item.get("workflow_run")
        if not isinstance(workflow_run, dict):
            return False
        if current_run_id and str(workflow_run.get("id", "")) == current_run_id:
            return False
        if ref_name and str(workflow_run.get("head_branch", "")) != ref_name:
            return False
        event_name = str(workflow_run.get("event", "")).strip()
        if event_name and event_name not in {"push", "workflow_dispatch"}:
            return False
        return True

    artifact_candidates = [
        item for item in artifacts if _artifact_is_candidate(item)
    ]
    if not artifact_candidates:
        typer.echo(
            "No reusable same-branch dataflow-report artifact found; continuing without checkpoint."
        )
        return 0
    chunk_prefix = f"{state_name}.chunks/"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for artifact in artifact_candidates:
        download_url = str(artifact.get("archive_download_url", "") or "")
        try:
            archive_bytes = _download_artifact_archive_bytes(
                download_url=download_url,
                headers=headers,
                urlopen_fn=urlopen_fn,
                no_redirect_open_fn=no_redirect_open_fn,
                follow_redirect_open_fn=follow_redirect_open_fn,
            )
            checkpoint_member: str | None = None
            chunk_members: list[str] = []
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                names = [name for name in zf.namelist() if not name.endswith("/")]
                for name in names:
                    base = name.split("/", 1)[-1]
                    if base == state_name:
                        checkpoint_member = name
                    elif base.startswith(chunk_prefix):
                        chunk_members.append(name)
                if checkpoint_member is None:
                    continue
                checkpoint_bytes = zf.read(checkpoint_member)
                if _state_requires_chunk_artifacts(
                    checkpoint_bytes=checkpoint_bytes
                ) and not chunk_members:
                    continue
                checkpoint_output = output_dir / state_name
                chunk_output_dir = output_dir / f"{state_name}.chunks"
                if checkpoint_output.exists():
                    checkpoint_output.unlink()
                if chunk_output_dir.exists():
                    for existing in chunk_output_dir.glob("*"):
                        if existing.is_file():
                            existing.unlink()
                checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_output.write_bytes(checkpoint_bytes)
                restored = 1
                for name in chunk_members:
                    base = name.split("/", 1)[-1]
                    destination = output_dir / base
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(zf.read(name))
                    restored += 1
                typer.echo(
                    f"Restored {restored} ASPF state artifact file(s) from prior run."
                )
                return 0
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        typer.echo(
            f"Unable to restore ASPF state from prior artifacts ({last_error}); continuing without checkpoint."
        )
        return 0
    typer.echo(
        "Prior artifacts did not include usable ASPF state files; continuing without restore."
    )
    return 0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        _ = (req, fp, code, msg, headers, newurl)
        return None


def _download_artifact_archive_bytes(
    *,
    download_url: str,
    headers: Mapping[str, str],
    urlopen_fn: Callable[..., object] = urllib.request.urlopen,
    no_redirect_open_fn: Callable[..., object] | None = None,
    follow_redirect_open_fn: Callable[..., object] | None = None,
) -> bytes:
    req_zip = urllib.request.Request(download_url, headers=dict(headers))
    if urlopen_fn is not urllib.request.urlopen:
        with urlopen_fn(req_zip, timeout=60) as response:
            return response.read()
    if no_redirect_open_fn is None:
        no_redirect_open_fn = urllib.request.build_opener(_NoRedirectHandler()).open
    if follow_redirect_open_fn is None:
        follow_redirect_open_fn = urllib.request.urlopen
    try:
        with no_redirect_open_fn(req_zip, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        redirect_url = str(exc.headers.get("Location", "") or "")
        if not redirect_url:
            raise
        follow_headers: dict[str, str] = {}
        redirect_host = (urllib.parse.urlparse(redirect_url).hostname or "").lower()
        if redirect_host.endswith("github.com"):
            follow_headers = dict(headers)
        follow_req = urllib.request.Request(redirect_url, headers=follow_headers)
        with follow_redirect_open_fn(follow_req, timeout=60) as response:
            return response.read()


def _state_requires_chunk_artifacts(*, checkpoint_bytes: bytes) -> bool:
    try:
        payload = json.loads(checkpoint_bytes.decode("utf-8"))
    except Exception:
        return False
    if not isinstance(payload, Mapping):
        return False
    collection_resume = payload.get("collection_resume")
    if not isinstance(collection_resume, Mapping):
        return False
    analysis_index_resume = collection_resume.get("analysis_index_resume")
    if not isinstance(analysis_index_resume, Mapping):
        return False
    state_ref = analysis_index_resume.get("state_ref")
    return isinstance(state_ref, str) and bool(state_ref.strip())


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


def _run_synth(
    *,
    paths: List[Path] | None,
    root: Path,
    out_dir: Path,
    no_timestamp: bool,
    config: Optional[Path],
    exclude: Optional[List[str]],
    filter_bundle: DataflowFilterBundle | None,
    allow_external: Optional[bool],
    strictness: Optional[str],
    no_recursive: bool,
    max_components: int,
    type_audit_report: bool,
    type_audit_max: int,
    synthesis_max_tier: int,
    synthesis_min_bundle_size: int,
    synthesis_allow_singletons: bool,
    synthesis_protocols_kind: str,
    refactor_plan: bool,
    fail_on_violations: bool,
    aspf_trace_json: Path | None = None,
    aspf_import_trace: list[Path] | None = None,
    aspf_equivalence_against: list[Path] | None = None,
    aspf_opportunities_json: Path | None = None,
    aspf_state_json: Path | None = None,
    aspf_import_state: list[Path] | None = None,
    aspf_delta_jsonl: Path | None = None,
    aspf_semantic_surface: list[str] | None = None,
    runner: Runner = run_command,
) -> tuple[JSONObject, dict[str, Path], Path | None]:
    check_deadline()
    resolved_filter_bundle = filter_bundle or DataflowFilterBundle(None, None)
    if not paths:
        paths = [Path(".")]
    exclude_dirs: list[str] | None = None
    if exclude is not None:
        exclude_dirs = []
        for entry in exclude:
            check_deadline()
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list, transparent_list = resolved_filter_bundle.to_payload_lists()
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    if synthesis_protocols_kind not in {"dataclass", "protocol", "contextvar"}:
        raise typer.BadParameter(
            "synthesis-protocols-kind must be 'dataclass', 'protocol', or 'contextvar'"
        )

    output_root = out_dir
    timestamp = None
    if not no_timestamp:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = out_dir / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "LATEST.txt").write_text(timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "dataflow_report.md"
    dot_path = output_root / "dataflow_graph.dot"
    plan_path = output_root / "synthesis_plan.json"
    protocol_path = output_root / "protocol_stubs.py"
    refactor_plan_path = output_root / "refactor_plan.json"
    fingerprint_synth_path = output_root / "fingerprint_synth.json"
    fingerprint_provenance_path = output_root / "fingerprint_provenance.json"
    fingerprint_coherence_path = output_root / "fingerprint_coherence.json"
    fingerprint_rewrite_plans_path = output_root / "fingerprint_rewrite_plans.json"
    fingerprint_exception_obligations_path = (
        output_root / "fingerprint_exception_obligations.json"
    )
    fingerprint_handledness_path = output_root / "fingerprint_handledness.json"

    payload: JSONObject = {
        "paths": [str(p) for p in paths],
        "root": str(root),
        "config": str(config) if config is not None else None,
        "report": str(report_path),
        "dot": str(dot_path),
        "fail_on_violations": fail_on_violations,
        "no_recursive": no_recursive,
        "max_components": max_components,
        "type_audit_report": type_audit_report,
        "type_audit_max": type_audit_max,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "synthesis_plan": str(plan_path),
        "synthesis_report": True,
        "synthesis_protocols": str(protocol_path),
        "synthesis_protocols_kind": synthesis_protocols_kind,
        "synthesis_max_tier": synthesis_max_tier,
        "synthesis_min_bundle_size": synthesis_min_bundle_size,
        "synthesis_allow_singletons": synthesis_allow_singletons,
        "refactor_plan": refactor_plan,
        "refactor_plan_json": str(refactor_plan_path) if refactor_plan else None,
        "fingerprint_synth_json": str(fingerprint_synth_path),
        "fingerprint_provenance_json": str(fingerprint_provenance_path),
        "fingerprint_coherence_json": str(fingerprint_coherence_path),
        "fingerprint_rewrite_plans_json": str(fingerprint_rewrite_plans_path),
        "fingerprint_exception_obligations_json": str(
            fingerprint_exception_obligations_path
        ),
        "fingerprint_handledness_json": str(fingerprint_handledness_path),
        "aspf_trace_json": str(aspf_trace_json) if aspf_trace_json is not None else None,
        "aspf_import_trace": [str(path) for path in (aspf_import_trace or [])],
        "aspf_equivalence_against": [
            str(path) for path in (aspf_equivalence_against or [])
        ],
        "aspf_opportunities_json": (
            str(aspf_opportunities_json) if aspf_opportunities_json is not None else None
        ),
        "aspf_state_json": str(aspf_state_json) if aspf_state_json is not None else None,
        "aspf_import_state": [str(path) for path in (aspf_import_state or [])],
        "aspf_delta_jsonl": str(aspf_delta_jsonl) if aspf_delta_jsonl is not None else None,
        "aspf_semantic_surface": [
            str(surface) for surface in (aspf_semantic_surface or [])
        ],
    }
    result = dispatch_command(
        command=DATAFLOW_COMMAND,
        payload=payload,
        root=root,
        runner=runner,
    )
    paths_out = {
        "report": report_path,
        "dot": dot_path,
        "plan": plan_path,
        "protocol": protocol_path,
        "refactor": refactor_plan_path,
        "fingerprint_synth": fingerprint_synth_path,
        "fingerprint_provenance": fingerprint_provenance_path,
        "fingerprint_coherence": fingerprint_coherence_path,
        "fingerprint_rewrite_plans": fingerprint_rewrite_plans_path,
        "fingerprint_exception_obligations": fingerprint_exception_obligations_path,
        "fingerprint_handledness": fingerprint_handledness_path,
        "output_root": output_root,
    }
    return result, paths_out, timestamp


@app.command("synth")
def synth(
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    out_dir: Path = typer.Option(Path("artifacts/synthesis"), "--out-dir"),
    no_timestamp: bool = typer.Option(False, "--no-timestamp"),
    config: Optional[Path] = typer.Option(None, "--config"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params_csv: Optional[str] = typer.Option(None, "--ignore-params"),
    transparent_decorators_csv: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    no_recursive: bool = typer.Option(False, "--no-recursive"),
    max_components: int = typer.Option(10, "--max-components"),
    type_audit_report: bool = typer.Option(
        True, "--type-audit-report/--no-type-audit-report"
    ),
    type_audit_max: int = typer.Option(50, "--type-audit-max"),
    synthesis_max_tier: int = typer.Option(2, "--synthesis-max-tier"),
    synthesis_min_bundle_size: int = typer.Option(2, "--synthesis-min-bundle-size"),
    synthesis_allow_singletons: bool = typer.Option(
        False, "--synthesis-allow-singletons"
    ),
    synthesis_protocols_kind: str = typer.Option(
        "dataclass", "--synthesis-protocols-kind"
    ),
    refactor_plan: bool = typer.Option(True, "--refactor-plan/--no-refactor-plan"),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
    aspf_trace_json: Optional[Path] = typer.Option(None, "--aspf-trace-json"),
    aspf_import_trace: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-trace",
    ),
    aspf_equivalence_against: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-equivalence-against",
    ),
    aspf_opportunities_json: Optional[Path] = typer.Option(
        None,
        "--aspf-opportunities-json",
    ),
    aspf_state_json: Optional[Path] = typer.Option(
        None,
        "--aspf-state-json",
    ),
    aspf_delta_jsonl: Optional[Path] = typer.Option(
        None,
        "--aspf-delta-jsonl",
    ),
    aspf_import_state: Optional[List[Path]] = typer.Option(
        None,
        "--aspf-import-state",
    ),
    aspf_semantic_surface: Optional[List[str]] = typer.Option(
        None,
        "--aspf-semantic-surface",
    ),
) -> None:
    """Run the dataflow audit and emit synthesis outputs (prototype)."""
    with _cli_deadline_scope():
        filter_bundle = DataflowFilterBundle(
            ignore_params_csv=ignore_params_csv,
            transparent_decorators_csv=transparent_decorators_csv,
        )
        result, paths_out, timestamp = _run_synth(
            paths=paths,
            root=root,
            out_dir=out_dir,
            no_timestamp=no_timestamp,
            config=config,
            exclude=exclude,
            filter_bundle=filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            no_recursive=no_recursive,
            max_components=max_components,
            type_audit_report=type_audit_report,
            type_audit_max=type_audit_max,
            synthesis_max_tier=synthesis_max_tier,
            synthesis_min_bundle_size=synthesis_min_bundle_size,
            synthesis_allow_singletons=synthesis_allow_singletons,
            synthesis_protocols_kind=synthesis_protocols_kind,
            refactor_plan=refactor_plan,
            fail_on_violations=fail_on_violations,
            aspf_trace_json=aspf_trace_json,
            aspf_import_trace=aspf_import_trace,
            aspf_equivalence_against=aspf_equivalence_against,
            aspf_opportunities_json=aspf_opportunities_json,
            aspf_state_json=aspf_state_json,
            aspf_import_state=aspf_import_state,
            aspf_delta_jsonl=aspf_delta_jsonl,
            aspf_semantic_surface=aspf_semantic_surface,
        )
        _emit_synth_outputs(
            paths_out=paths_out,
            timestamp=timestamp,
            refactor_plan=refactor_plan,
        )
        raise typer.Exit(code=int(result.get("exit_code", 0)))


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
    try:
        return int(main_fn(argv))
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return int(code)
        return 1


_TOOLING_NO_ARG_RUNNERS: dict[str, Callable[[], int]] = {
    "delta-advisory-telemetry": tooling_delta_advisory.telemetry_main,
    "docflow-delta-emit": tooling_docflow_delta_emit.main,
}
_TOOLING_ARGV_RUNNERS: dict[str, Callable[[list[str] | None], int]] = {
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
    previous_no_arg = dict(_TOOLING_NO_ARG_RUNNERS)
    previous_with_argv = dict(_TOOLING_ARGV_RUNNERS)
    if isinstance(no_arg, Mapping):
        _TOOLING_NO_ARG_RUNNERS.update(
            {str(key): value for key, value in no_arg.items() if callable(value)}
        )
    if isinstance(with_argv, Mapping):
        _TOOLING_ARGV_RUNNERS.update(
            {str(key): value for key, value in with_argv.items() if callable(value)}
        )
    try:
        yield
    finally:
        _TOOLING_NO_ARG_RUNNERS.clear()
        _TOOLING_NO_ARG_RUNNERS.update(previous_no_arg)
        _TOOLING_ARGV_RUNNERS.clear()
        _TOOLING_ARGV_RUNNERS.update(previous_with_argv)


def _run_tooling_no_arg(command_name: str) -> int:
    runner = _TOOLING_NO_ARG_RUNNERS[command_name]
    with _cli_deadline_scope():
        return int(runner())


def _run_tooling_with_argv(command_name: str, argv: list[str]) -> int:
    runner = _TOOLING_ARGV_RUNNERS[command_name]
    with _cli_deadline_scope():
        return _invoke_argparse_command(runner, argv)


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


@app.command("delta-advisory-telemetry")
def delta_advisory_telemetry() -> None:
    """Emit non-blocking advisory telemetry artifacts."""
    raise typer.Exit(code=_run_tooling_no_arg("delta-advisory-telemetry"))


@app.command("docflow-delta-emit")
def docflow_delta_emit() -> None:
    """Emit docflow compliance delta through the gabion CLI."""
    raise typer.Exit(code=_run_tooling_no_arg("docflow-delta-emit"))


@app.command(
    "impact-select-tests",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def impact_select_tests(ctx: typer.Context) -> None:
    """Select impacted tests from diffs and evidence index."""
    raise typer.Exit(
        code=_run_tooling_with_argv(
            "impact-select-tests",
            list(ctx.args),
        )
    )


@app.command(
    "run-dataflow-stage",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_dataflow_stage(ctx: typer.Context) -> None:
    """Run a single dataflow stage with CI-aligned outputs."""
    raise typer.Exit(
        code=_run_tooling_with_argv(
            "run-dataflow-stage",
            list(ctx.args),
        )
    )


@app.command(
    "ambiguity-contract-gate",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def ambiguity_contract_gate(ctx: typer.Context) -> None:
    """Run ambiguity-contract policy gate for deterministic-core surfaces."""
    raise typer.Exit(
        code=_run_tooling_with_argv(
            "ambiguity-contract-gate",
            list(ctx.args),
        )
    )


@app.command(
    "normative-symdiff",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def normative_symdiff(ctx: typer.Context) -> None:
    """Compute a normative-docs ↔ code/tooling symmetric-difference report."""
    raise typer.Exit(
        code=_run_tooling_with_argv(
            "normative-symdiff",
            list(ctx.args),
        )
    )


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
    protocol_name: Optional[str] = typer.Option(None, "--protocol-name"),
    bundle: Optional[List[str]] = typer.Option(None, "--bundle"),
    field: Optional[List[str]] = typer.Option(
        None,
        "--field",
        help="Field spec in 'name:type' form (repeatable).",
    ),
    target_path: Optional[Path] = typer.Option(None, "--target-path"),
    target_functions: Optional[List[str]] = typer.Option(None, "--target-function"),
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
            protocol_name=protocol_name,
            bundle=bundle,
            field=field,
            target_path=target_path,
            target_functions=target_functions,
            compatibility_shim=compatibility_shim,
            compatibility_shim_warnings=compatibility_shim_warnings,
            compatibility_shim_overloads=compatibility_shim_overloads,
            ambient_rewrite=ambient_rewrite,
            rationale=rationale,
        )


def _run_refactor_protocol(
    *,
    input_path: Optional[Path],
    output_path: Optional[Path],
    protocol_name: Optional[str],
    bundle: Optional[List[str]],
    field: Optional[List[str]],
    target_path: Optional[Path],
    target_functions: Optional[List[str]],
    compatibility_shim: bool,
    compatibility_shim_warnings: bool,
    compatibility_shim_overloads: bool,
    ambient_rewrite: bool,
    rationale: Optional[str],
    runner: Runner = run_command,
) -> None:
    """Generate protocol refactor edits from a JSON payload (prototype)."""
    input_payload: JSONObject | None = None
    if input_path is not None:
        try:
            loaded = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
        if not isinstance(loaded, dict):
            raise typer.BadParameter("Refactor payload must be a JSON object.")
        input_payload = loaded
    payload = build_refactor_payload(
        input_payload=input_payload,
        protocol_name=protocol_name,
        bundle=bundle,
        field=field,
        target_path=target_path,
        target_functions=target_functions,
        compatibility_shim=compatibility_shim,
        compatibility_shim_warnings=compatibility_shim_warnings,
        compatibility_shim_overloads=compatibility_shim_overloads,
        ambient_rewrite=ambient_rewrite,
        rationale=rationale,
    )
    result = dispatch_command(
        command=REFACTOR_COMMAND,
        payload=payload,
        root=None,
        runner=runner,
    )
    normalized = RefactorProtocolResponseDTO.model_validate(result).model_dump()
    output = json.dumps(normalized, indent=2, sort_keys=False)
    if output_path is None:
        typer.echo(output)
    else:
        _write_text_to_target(output_path, output)
