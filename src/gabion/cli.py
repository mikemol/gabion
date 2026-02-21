from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import ExitStack, contextmanager
from collections import OrderedDict
from typing import Callable, Generator, List, Mapping, MutableMapping, Optional, TypeAlias
import argparse
import importlib.util
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
from gabion.analysis.aspf import Forest
from gabion.analysis.timeout_context import (
    Deadline,
    check_deadline,
    deadline_loop_iter,
    deadline_clock_scope,
    deadline_scope,
    forest_scope,
    render_deadline_profile_markdown,
)
from gabion.deadline_clock import GasMeter

DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
REFACTOR_COMMAND = "gabion.refactorProtocol"
STRUCTURE_DIFF_COMMAND = "gabion.structureDiff"
STRUCTURE_REUSE_COMMAND = "gabion.structureReuse"
DECISION_DIFF_COMMAND = "gabion.decisionDiff"
IMPACT_COMMAND = "gabion.impactQuery"
from gabion.lsp_client import (
    CommandRequest,
    run_command,
    run_command_direct,
    _env_timeout_ticks,
    _has_env_timeout,
)
from gabion.plan import (
    ExecutionPlan,
    ExecutionPlanObligations,
    ExecutionPlanPolicyMetadata,
)
from gabion.json_types import JSONObject
from gabion.order_contract import ordered_or_sorted
from gabion.schema import (
    DataflowAuditResponseDTO,
    DecisionDiffResponseDTO,
    RefactorProtocolResponseDTO,
    StructureDiffResponseDTO,
    StructureReuseResponseDTO,
    SynthesisPlanResponseDTO,
)
app = typer.Typer(add_completion=False)
Runner: TypeAlias = Callable[..., JSONObject]
DEFAULT_RUNNER: Runner = run_command

_LINT_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$")

_DEFAULT_TIMEOUT_TICKS = 100
_DEFAULT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_CHECK_REPORT_REL_PATH = Path("artifacts/audit_reports/dataflow_report.md")
_DEFAULT_TIMEOUT_PROGRESS_REPORT_REL_PATH = Path(
    "artifacts/audit_reports/timeout_progress.md"
)
_SPPF_GH_REF_RE = re.compile(r"\bGH-(\d+)\b", re.IGNORECASE)
_SPPF_KEYWORD_REF_RE = re.compile(
    r"\b(?:Closes|Fixes|Resolves|Refs)\s+#(\d+)\b", re.IGNORECASE
)

_LSP_PROGRESS_NOTIFICATION_METHOD = "$/progress"
_LSP_PROGRESS_TOKEN = "gabion.dataflowAudit/progress-v1"
_STDOUT_ALIAS = "-"
_STDOUT_PATH = "/dev/stdout"


@dataclass(frozen=True)
class SppfSyncCommitInfo:
    sha: str
    subject: str
    body: str


def _cli_timeout_ticks() -> tuple[int, int]:
    if _has_env_timeout():
        return _env_timeout_ticks()
    return _DEFAULT_TIMEOUT_TICKS, _DEFAULT_TIMEOUT_TICK_NS


def _cli_deadline() -> Deadline:
    ticks, tick_ns = _cli_timeout_ticks()
    return Deadline.from_timeout_ticks(ticks, tick_ns)


def _resolve_check_report_path(report: Path | None, *, root: Path) -> Path:
    if report is not None:
        return report
    return root / _DEFAULT_CHECK_REPORT_REL_PATH


@contextmanager
def _cli_deadline_scope():
    ticks, _tick_ns = _cli_timeout_ticks()
    with ExitStack() as stack:
        stack.enter_context(forest_scope(Forest()))
        stack.enter_context(deadline_scope(_cli_deadline()))
        stack.enter_context(deadline_clock_scope(GasMeter(limit=int(ticks))))
        yield


@dataclass(frozen=True)
class CheckArtifactFlags:
    emit_test_obsolescence: bool
    emit_test_evidence_suggestions: bool
    emit_call_clusters: bool
    emit_call_cluster_consolidation: bool
    emit_test_annotation_drift: bool
    emit_semantic_coverage_map: bool = False


@dataclass(frozen=True)
class CheckPolicyFlags:
    fail_on_violations: bool
    fail_on_type_ambiguities: bool
    lint: bool


@dataclass(frozen=True)
class DataflowPayloadCommonOptions:
    paths: list[Path]
    root: Path
    config: Path | None
    report: Path | None
    fail_on_violations: bool
    fail_on_type_ambiguities: bool
    baseline: Path | None
    baseline_write: bool | None
    decision_snapshot: Path | None
    exclude: list[str] | None
    filter_bundle: DataflowFilterBundle
    allow_external: bool | None
    strictness: str | None
    lint: bool
    resume_checkpoint: Path | None
    emit_timeout_progress_report: bool
    resume_on_timeout: int
    deadline_profile: bool = True


@dataclass(frozen=True)
class DataflowFilterBundle:
    ignore_params_csv: str | None
    transparent_decorators_csv: str | None

    def to_payload_lists(self) -> tuple[list[str] | None, list[str] | None]:
        ignore_list = (
            _split_csv(self.ignore_params_csv)
            if self.ignore_params_csv is not None
            else None
        )
        transparent_list = (
            _split_csv(self.transparent_decorators_csv)
            if self.transparent_decorators_csv is not None
            else None
        )
        return ignore_list, transparent_list


@dataclass(frozen=True)
class CheckDeltaOptions:
    emit_test_obsolescence_state: bool
    test_obsolescence_state: Path | None
    emit_test_obsolescence_delta: bool
    test_annotation_drift_state: Path | None
    emit_test_annotation_drift_delta: bool
    write_test_annotation_drift_baseline: bool
    write_test_obsolescence_baseline: bool
    emit_ambiguity_delta: bool
    emit_ambiguity_state: bool
    ambiguity_state: Path | None
    write_ambiguity_baseline: bool
    semantic_coverage_mapping: Path | None = None

    def validate(self) -> None:
        if self.emit_test_obsolescence_delta and self.write_test_obsolescence_baseline:
            raise typer.BadParameter(
                "Use --emit-test-obsolescence-delta or --write-test-obsolescence-baseline, not both."
            )
        if self.emit_test_obsolescence_state and self.test_obsolescence_state is not None:
            raise typer.BadParameter(
                "Use --emit-test-obsolescence-state or --test-obsolescence-state, not both."
            )
        if (
            self.emit_test_annotation_drift_delta
            and self.write_test_annotation_drift_baseline
        ):
            raise typer.BadParameter(
                "Use --emit-test-annotation-drift-delta or --write-test-annotation-drift-baseline, not both."
            )
        if self.emit_ambiguity_delta and self.write_ambiguity_baseline:
            raise typer.BadParameter(
                "Use --emit-ambiguity-delta or --write-ambiguity-baseline, not both."
            )
        if self.emit_ambiguity_state and self.ambiguity_state is not None:
            raise typer.BadParameter(
                "Use --emit-ambiguity-state or --ambiguity-state, not both."
            )

    def to_payload(self) -> JSONObject:
        return {
            "emit_test_obsolescence_state": self.emit_test_obsolescence_state,
            "test_obsolescence_state": str(self.test_obsolescence_state)
            if self.test_obsolescence_state is not None
            else None,
            "emit_test_obsolescence_delta": self.emit_test_obsolescence_delta,
            "test_annotation_drift_state": str(self.test_annotation_drift_state)
            if self.test_annotation_drift_state is not None
            else None,
            "emit_test_annotation_drift_delta": self.emit_test_annotation_drift_delta,
            "write_test_annotation_drift_baseline": self.write_test_annotation_drift_baseline,
            "semantic_coverage_mapping": str(self.semantic_coverage_mapping)
            if self.semantic_coverage_mapping is not None
            else None,
            "write_test_obsolescence_baseline": self.write_test_obsolescence_baseline,
            "emit_ambiguity_delta": self.emit_ambiguity_delta,
            "emit_ambiguity_state": self.emit_ambiguity_state,
            "ambiguity_state": str(self.ambiguity_state)
            if self.ambiguity_state is not None
            else None,
            "write_ambiguity_baseline": self.write_ambiguity_baseline,
        }


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


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    issues = set(match.group(1) for match in _SPPF_GH_REF_RE.finditer(text))
    issues.update(match.group(1) for match in _SPPF_KEYWORD_REF_RE.finditer(text))
    return issues


def _issue_ids_from_sppf_commits(commits: list[SppfSyncCommitInfo]) -> set[str]:
    issues: set[str] = set()
    for commit in commits:
        check_deadline()
        issues.update(_extract_sppf_issue_ids(commit.subject))
        issues.update(_extract_sppf_issue_ids(commit.body))
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

    issue_ids = ordered_or_sorted(
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


def _normalize_optional_output_target(target: object) -> str | None:
    if target is None:
        return None
    text = str(target).strip()
    if not text:
        return None
    return _normalize_output_target(text)


def _write_lint_jsonl(target: str, entries: list[dict[str, object]]) -> None:
    payload = "\n".join(json.dumps(entry, sort_keys=True) for entry in entries)
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
    duplicate_codes = sorted(
        code for code, count in rule_counts.items() if int(count) > 1
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
    payload = json.dumps(sarif, indent=2, sort_keys=True)
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
    profile_json_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
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
    if isinstance(progress_ticks_per_ns, (int, float)):
        lines.append(f"- `ticks_per_ns`: `{float(progress_ticks_per_ns):.9f}`")
    if isinstance(deadline_profile, Mapping):
        if not isinstance(progress_ticks_per_ns, (int, float)):
            ticks_per_ns = deadline_profile.get("ticks_per_ns")
            if isinstance(ticks_per_ns, (int, float)):
                lines.append(f"- `ticks_per_ns`: `{float(ticks_per_ns):.9f}`")
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


def _emit_timeout_progress_artifacts(
    result: Mapping[str, object],
    *,
    root: Path,
) -> None:
    timeout_context = result.get("timeout_context")
    if not isinstance(timeout_context, Mapping):
        return
    progress = timeout_context.get("progress")
    if not isinstance(progress, Mapping):
        return
    profile = timeout_context.get("deadline_profile")
    profile_mapping = profile if isinstance(profile, Mapping) else None
    progress_md_path = root / _DEFAULT_TIMEOUT_PROGRESS_REPORT_REL_PATH
    progress_json_path = progress_md_path.with_suffix(".json")
    progress_md_path.parent.mkdir(parents=True, exist_ok=True)
    payload: JSONObject = {
        "analysis_state": str(result.get("analysis_state", "")),
        "progress": {str(key): progress[key] for key in progress},
    }
    progress_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with _cli_deadline_scope():
        rendered = _render_timeout_progress_markdown(
            analysis_state=str(result.get("analysis_state", "")),
            progress=progress,
            deadline_profile=profile_mapping,
        )
    progress_md_path.write_text(
        rendered + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Wrote timeout progress JSON: {progress_json_path}")
    typer.echo(f"Wrote timeout progress markdown: {progress_md_path}")


def _build_dataflow_payload_common(
    *,
    options: DataflowPayloadCommonOptions,
) -> JSONObject:
    # dataflow-bundle: filter_bundle
    # dataflow-bundle: deadline_profile, emit_timeout_progress_report
    exclude_dirs = _split_csv_entries(options.exclude) if options.exclude is not None else None
    ignore_list, transparent_list = options.filter_bundle.to_payload_lists()
    payload: JSONObject = {
        "paths": [str(p) for p in options.paths],
        "root": str(options.root),
        "config": str(options.config) if options.config is not None else None,
        "report": str(options.report) if options.report is not None else None,
        "fail_on_violations": options.fail_on_violations,
        "fail_on_type_ambiguities": options.fail_on_type_ambiguities,
        "baseline": str(options.baseline) if options.baseline is not None else None,
        "baseline_write": options.baseline_write,
        "decision_snapshot": str(options.decision_snapshot)
        if options.decision_snapshot is not None
        else None,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": options.allow_external,
        "strictness": options.strictness,
        "lint": options.lint,
        "resume_checkpoint": str(options.resume_checkpoint)
        if options.resume_checkpoint is not None
        else None,
        "emit_timeout_progress_report": bool(options.emit_timeout_progress_report),
        "resume_on_timeout": int(options.resume_on_timeout),
        "deadline_profile": bool(options.deadline_profile),
    }
    return payload


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
    resume_checkpoint: Optional[Path] = None,
    emit_timeout_progress_report: bool = False,
    resume_on_timeout: int = 0,
    analysis_tick_limit: int | None = None,
) -> JSONObject:
    if filter_bundle is None:
        filter_bundle = DataflowFilterBundle(None, None)
    if not paths:
        paths = [Path(".")]
    delta_options.validate()
    baseline_write_value = bool(baseline is not None and baseline_write)
    payload = _build_dataflow_payload_common(
        options=DataflowPayloadCommonOptions(
            paths=paths,
            root=root,
            config=config,
            report=report,
            fail_on_violations=fail_on_violations,
            fail_on_type_ambiguities=fail_on_type_ambiguities,
            baseline=baseline,
            baseline_write=baseline_write_value,
            decision_snapshot=decision_snapshot,
            exclude=exclude,
            filter_bundle=filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            lint=lint,
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
        )
    )
    payload.update(
        {
            "emit_test_obsolescence": artifact_flags.emit_test_obsolescence,
            "emit_test_evidence_suggestions": artifact_flags.emit_test_evidence_suggestions,
            "emit_call_clusters": artifact_flags.emit_call_clusters,
            "emit_call_cluster_consolidation": artifact_flags.emit_call_cluster_consolidation,
            "emit_test_annotation_drift": artifact_flags.emit_test_annotation_drift,
            "emit_semantic_coverage_map": artifact_flags.emit_semantic_coverage_map,
            "type_audit": True if fail_on_type_ambiguities else None,
            "semantic_coverage_mapping": str(delta_options.semantic_coverage_mapping)
            if delta_options.semantic_coverage_mapping is not None
            else None,
            "analysis_tick_limit": int(analysis_tick_limit)
            if analysis_tick_limit is not None
            else None,
        }
    )
    payload.update(delta_options.to_payload())
    return payload




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
) -> ExecutionPlanRequest:
    operations = [DATAFLOW_COMMAND, "gabion.check"]
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
    structure_tree_target = _normalize_optional_output_target(opts.emit_structure_tree)
    structure_metrics_target = _normalize_optional_output_target(
        opts.emit_structure_metrics
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
            resume_checkpoint=Path(opts.resume_checkpoint)
            if opts.resume_checkpoint
            else None,
            emit_timeout_progress_report=bool(opts.emit_timeout_progress_report),
            resume_on_timeout=max(int(opts.resume_on_timeout), 0),
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
    ticks, tick_ns = _cli_timeout_ticks()
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        payload = dict(payload)
        payload["analysis_timeout_ticks"] = int(ticks)
        payload["analysis_timeout_tick_ns"] = int(tick_ns)
    if execution_plan_request is not None:
        payload = dict(payload)
        execution_plan_payload = execution_plan_request.to_payload()
        execution_plan_inputs = execution_plan_payload.get("inputs")
        if isinstance(execution_plan_inputs, Mapping):
            merged_inputs = dict(execution_plan_inputs)
            merged_inputs.update(payload)
            execution_plan_payload["inputs"] = merged_inputs
        deadline_metadata = execution_plan_payload.get("policy_metadata")
        if isinstance(deadline_metadata, Mapping):
            policy_metadata = dict(deadline_metadata)
            deadline = policy_metadata.get("deadline")
            deadline_payload = dict(deadline) if isinstance(deadline, Mapping) else {}
            deadline_payload["analysis_timeout_ticks"] = int(
                payload.get("analysis_timeout_ticks") or 0
            )
            deadline_payload["analysis_timeout_tick_ns"] = int(
                payload.get("analysis_timeout_tick_ns") or 0
            )
            policy_metadata["deadline"] = deadline_payload
            execution_plan_payload["policy_metadata"] = policy_metadata
        payload["execution_plan_request"] = execution_plan_payload
    request = CommandRequest(command, [payload])
    resolved = runner
    if runner is run_command:
        flag = os.getenv("GABION_DIRECT_RUN", "").strip().lower()
        if flag in {"1", "true", "yes", "on"}:
            resolved = run_command_direct
    if resolved is run_command:
        factory = process_factory or subprocess.Popen
        return resolved(
            request,
            root=root,
            timeout_ticks=ticks,
            timeout_tick_ns=tick_ns,
            process_factory=factory,
            notification_callback=notification_callback,
        )
    if resolved is run_command_direct:
        return resolved(
            request,
            root=root,
            notification_callback=notification_callback,
        )
    if notification_callback is not None:
        try:
            params = inspect.signature(resolved).parameters
        except (TypeError, ValueError):
            params = {}
        if "notification_callback" in params or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        ):
            return resolved(
                request,
                root=root,
                notification_callback=notification_callback,
            )
    return resolved(request, root=root)


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
    resume_checkpoint: Optional[Path] = None,
    emit_timeout_progress_report: bool = False,
    resume_on_timeout: int = 0,
    analysis_tick_limit: int | None = None,
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
        resume_checkpoint=resume_checkpoint,
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
        analysis_tick_limit=analysis_tick_limit,
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
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
    emit_timeout_profile_artifacts_fn: Callable[..., None] = _emit_timeout_profile_artifacts,
    emit_timeout_progress_artifacts_fn: Callable[..., None] = _emit_timeout_progress_artifacts,
    echo_fn: Callable[[str], None] = typer.echo,
) -> JSONObject:
    attempt = 0
    result: JSONObject = {}
    while True:
        with _cli_deadline_scope():
            check_deadline()
            result = run_once()
        if result.get("timeout") is not True:
            return result
        emit_timeout_profile_artifacts_fn(result, root=root)
        if emit_timeout_progress_report:
            emit_timeout_progress_artifacts_fn(result, root=root)
        if (
            attempt < max(int(resume_on_timeout), 0)
            and str(result.get("analysis_state", "")) == "timed_out_progress_resume"
        ):
            attempt += 1
            echo_fn(f"Retrying after timeout with progress ({attempt}/{resume_on_timeout})...")
            continue
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
                json.dumps(result["synthesis_plan"], indent=2, sort_keys=True),
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
                json.dumps(result["refactor_plan"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_synth_json)
            and "fingerprint_synth_registry" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_synth_registry"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_provenance_json)
            and "fingerprint_provenance" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_provenance"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.fingerprint_deadness_json) and "fingerprint_deadness" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_deadness"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.fingerprint_coherence_json) and "fingerprint_coherence" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_coherence"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_rewrite_plans_json)
            and "fingerprint_rewrite_plans" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_rewrite_plans"], indent=2, sort_keys=True),
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
                    sort_keys=True,
                ),
                ensure_trailing_newline=True,
            )
        if (
            _is_stdout_target(opts.fingerprint_handledness_json)
            and "fingerprint_handledness" in result
        ):
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["fingerprint_handledness"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.emit_structure_tree) and "structure_tree" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["structure_tree"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.emit_structure_metrics) and "structure_metrics" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["structure_metrics"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
            )
        if _is_stdout_target(opts.emit_decision_snapshot) and "decision_snapshot" in result:
            _write_text_to_target(
                _STDOUT_PATH,
                json.dumps(result["decision_snapshot"], indent=2, sort_keys=True),
                ensure_trailing_newline=True,
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
    resume_checkpoint: Optional[Path],
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
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
    if _param_is_command_line(ctx, "resume_checkpoint") and resume_checkpoint is not None:
        argv.extend(["--resume-checkpoint", str(resume_checkpoint)])
    if _param_is_command_line(ctx, "emit_timeout_progress_report") and emit_timeout_progress_report:
        argv.append("--emit-timeout-progress-report")
    if _param_is_command_line(ctx, "resume_on_timeout"):
        argv.extend(["--resume-on-timeout", str(int(resume_on_timeout))])
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
    resume_checkpoint: Optional[Path],
    emit_timeout_progress_report: bool,
    resume_on_timeout: int,
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
        resume_checkpoint=resume_checkpoint,
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
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


def _emit_resume_checkpoint_startup_line(
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
        "resume checkpoint detected... "
        f"path={checkpoint_path or '<none>'} "
        f"status={status or 'unknown'} "
        f"reused_files={reused_display}"
    )


def _resume_checkpoint_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    if str(notification.get("method", "") or "") != _LSP_PROGRESS_NOTIFICATION_METHOD:
        return None
    params = notification.get("params")
    if not isinstance(params, Mapping):
        return None
    if str(params.get("token", "") or "") != _LSP_PROGRESS_TOKEN:
        return None
    value = params.get("value")
    if not isinstance(value, Mapping):
        return None
    resume_checkpoint = value.get("resume_checkpoint")
    if not isinstance(resume_checkpoint, Mapping):
        return None
    checkpoint_path = str(resume_checkpoint.get("checkpoint_path", "") or "")
    status = str(resume_checkpoint.get("status", "") or "")
    reused_files = int(resume_checkpoint.get("reused_files", 0) or 0)
    total_files = int(resume_checkpoint.get("total_files", 0) or 0)
    return {
        "checkpoint_path": checkpoint_path,
        "status": status,
        "reused_files": reused_files,
        "total_files": total_files,
    }


def _checkpoint_intro_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    if str(notification.get("method", "") or "") != _LSP_PROGRESS_NOTIFICATION_METHOD:
        return None
    params = notification.get("params")
    if not isinstance(params, Mapping):
        return None
    if str(params.get("token", "") or "") != _LSP_PROGRESS_TOKEN:
        return None
    value = params.get("value")
    if not isinstance(value, Mapping):
        return None
    row = value.get("checkpoint_intro_timeline_row")
    if not isinstance(row, str) or not row:
        return None
    header = value.get("checkpoint_intro_timeline_header")
    return {
        "header": header if isinstance(header, str) else "",
        "row": row,
    }


def _phase_timeline_header_columns() -> list[str]:
    return [
        "ts_utc",
        "event_seq",
        "event_kind",
        "phase",
        "analysis_state",
        "classification",
        "progress_marker",
        "primary",
        "files",
        "resume_checkpoint",
        "stale_for_s",
        "dimensions",
    ]


def _phase_timeline_header_block() -> str:
    header = _phase_timeline_header_columns()
    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join(["---"] * len(header)) + " |"
    return header_line + "\n" + separator_line


def _phase_progress_dimensions_summary(
    phase_progress_v2: Mapping[str, object] | None,
) -> str:
    if not isinstance(phase_progress_v2, Mapping):
        return ""
    raw_dimensions = phase_progress_v2.get("dimensions")
    if not isinstance(raw_dimensions, Mapping):
        return ""
    fragments: list[str] = []
    dim_names = ordered_or_sorted(
        (name for name in raw_dimensions if isinstance(name, str)),
        source="_phase_progress_dimensions_summary.dim_names",
    )
    for dim_name in dim_names:
        raw_payload = raw_dimensions.get(dim_name)
        if not isinstance(raw_payload, Mapping):
            continue
        raw_done = raw_payload.get("done")
        raw_total = raw_payload.get("total")
        if (
            isinstance(raw_done, int)
            and not isinstance(raw_done, bool)
            and isinstance(raw_total, int)
            and not isinstance(raw_total, bool)
        ):
            done = max(int(raw_done), 0)
            total = max(int(raw_total), 0)
            if total:
                done = min(done, total)
            fragments.append(f"{dim_name}={done}/{total}")
    return "; ".join(fragments)


def _phase_timeline_row_from_phase_progress(
    phase_progress: Mapping[str, object],
) -> str:
    ts_utc = str(phase_progress.get("ts_utc", "") or "")
    event_seq = phase_progress.get("event_seq")
    event_kind = str(phase_progress.get("event_kind", "") or "")
    phase = str(phase_progress.get("phase", "") or "")
    analysis_state = str(phase_progress.get("analysis_state", "") or "")
    classification = str(phase_progress.get("classification", "") or "")
    progress_marker = str(phase_progress.get("progress_marker", "") or "")
    phase_progress_v2 = (
        phase_progress.get("phase_progress_v2")
        if isinstance(phase_progress.get("phase_progress_v2"), Mapping)
        else None
    )
    primary_unit = ""
    primary_done: int | None = None
    primary_total: int | None = None
    if isinstance(phase_progress_v2, Mapping):
        primary_unit = str(phase_progress_v2.get("primary_unit", "") or "")
        raw_primary_done = phase_progress_v2.get("primary_done")
        raw_primary_total = phase_progress_v2.get("primary_total")
        if isinstance(raw_primary_done, int) and not isinstance(raw_primary_done, bool):
            primary_done = max(int(raw_primary_done), 0)
        if isinstance(raw_primary_total, int) and not isinstance(raw_primary_total, bool):
            primary_total = max(int(raw_primary_total), 0)
        if (
            primary_done is not None
            and primary_total is not None
            and primary_total > 0
            and primary_done > primary_total
        ):
            primary_done = primary_total
    if primary_done is None or primary_total is None:
        raw_work_done = phase_progress.get("work_done")
        raw_work_total = phase_progress.get("work_total")
        if isinstance(raw_work_done, int) and isinstance(raw_work_total, int):
            primary_done = max(int(raw_work_done), 0)
            primary_total = max(int(raw_work_total), 0)
            if primary_total:
                primary_done = min(primary_done, primary_total)
    primary = ""
    if primary_done is not None and primary_total is not None:
        primary = f"{primary_done}/{primary_total}"
        if primary_unit:
            primary = f"{primary} {primary_unit}"
    elif primary_unit:
        primary = primary_unit
    completed_files = phase_progress.get("completed_files")
    remaining_files = phase_progress.get("remaining_files")
    total_files = phase_progress.get("total_files")
    files = ""
    if (
        isinstance(completed_files, int)
        and isinstance(remaining_files, int)
        and isinstance(total_files, int)
    ):
        files = f"{completed_files}/{total_files} rem={remaining_files}"
    resume_checkpoint = ""
    raw_resume = phase_progress.get("resume_checkpoint")
    if isinstance(raw_resume, Mapping):
        checkpoint_path = str(raw_resume.get("checkpoint_path", "") or "")
        status = str(raw_resume.get("status", "") or "")
        raw_reused = raw_resume.get("reused_files")
        raw_resume_total = raw_resume.get("total_files")
        if isinstance(raw_reused, int) and isinstance(raw_resume_total, int):
            resume_checkpoint = (
                f"path={checkpoint_path or '<none>'} status={status or 'unknown'} "
                f"reused_files={raw_reused}/{raw_resume_total}"
            )
        else:
            resume_checkpoint = (
                f"path={checkpoint_path or '<none>'} status={status or 'unknown'} "
                "reused_files=unknown"
            )
    raw_stale_for_s = phase_progress.get("stale_for_s")
    stale_for_s = (
        f"{float(raw_stale_for_s):.1f}"
        if isinstance(raw_stale_for_s, (int, float))
        else ""
    )
    dimensions = _phase_progress_dimensions_summary(
        phase_progress_v2 if isinstance(phase_progress_v2, Mapping) else None
    )
    row = [
        ts_utc,
        event_seq if isinstance(event_seq, int) else "",
        event_kind,
        phase,
        analysis_state,
        classification,
        progress_marker,
        primary,
        files,
        resume_checkpoint,
        stale_for_s,
        dimensions,
    ]
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"


def _phase_timeline_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, str] | None:
    phase_progress = _phase_progress_from_progress_notification(notification)
    if not isinstance(phase_progress, Mapping):
        return None
    header_value = phase_progress.get("phase_timeline_header")
    row_value = phase_progress.get("phase_timeline_row")
    header = (
        str(header_value)
        if isinstance(header_value, str) and header_value
        else _phase_timeline_header_block()
    )
    row = (
        str(row_value)
        if isinstance(row_value, str) and row_value
        else _phase_timeline_row_from_phase_progress(phase_progress)
    )
    return {"header": header, "row": row}


def _phase_progress_from_progress_notification(
    notification: Mapping[str, object],
) -> dict[str, object] | None:
    if str(notification.get("method", "") or "") != _LSP_PROGRESS_NOTIFICATION_METHOD:
        return None
    params = notification.get("params")
    if not isinstance(params, Mapping):
        return None
    if str(params.get("token", "") or "") != _LSP_PROGRESS_TOKEN:
        return None
    value = params.get("value")
    if not isinstance(value, Mapping):
        return None
    phase = str(value.get("phase", "") or "")
    if not phase:
        return None
    raw_work_done = value.get("work_done")
    work_done = (
        int(raw_work_done)
        if isinstance(raw_work_done, int) and not isinstance(raw_work_done, bool)
        else None
    )
    raw_work_total = value.get("work_total")
    work_total = (
        int(raw_work_total)
        if isinstance(raw_work_total, int) and not isinstance(raw_work_total, bool)
        else None
    )
    raw_completed_files = value.get("completed_files")
    completed_files = (
        int(raw_completed_files)
        if isinstance(raw_completed_files, int) and not isinstance(raw_completed_files, bool)
        else None
    )
    raw_remaining_files = value.get("remaining_files")
    remaining_files = (
        int(raw_remaining_files)
        if isinstance(raw_remaining_files, int) and not isinstance(raw_remaining_files, bool)
        else None
    )
    raw_total_files = value.get("total_files")
    total_files = (
        int(raw_total_files)
        if isinstance(raw_total_files, int) and not isinstance(raw_total_files, bool)
        else None
    )
    analysis_state = str(value.get("analysis_state", "") or "")
    classification = str(value.get("classification", "") or "")
    event_kind = str(value.get("event_kind", "") or "")
    progress_marker = str(value.get("progress_marker", "") or "")
    raw_event_seq = value.get("event_seq")
    event_seq = (
        int(raw_event_seq)
        if isinstance(raw_event_seq, int) and not isinstance(raw_event_seq, bool)
        else None
    )
    raw_stale_for_s = value.get("stale_for_s")
    stale_for_s = (
        float(raw_stale_for_s)
        if isinstance(raw_stale_for_s, (int, float)) and not isinstance(raw_stale_for_s, bool)
        else None
    )
    phase_progress_v2 = value.get("phase_progress_v2")
    normalized_phase_progress_v2 = (
        {str(key): phase_progress_v2[key] for key in phase_progress_v2}
        if isinstance(phase_progress_v2, Mapping)
        else None
    )
    phase_timeline_header = value.get("phase_timeline_header")
    phase_timeline_row = value.get("phase_timeline_row")
    resume_checkpoint = value.get("resume_checkpoint")
    normalized_resume_checkpoint = (
        {str(key): resume_checkpoint[key] for key in resume_checkpoint}
        if isinstance(resume_checkpoint, Mapping)
        else None
    )
    done = bool(value.get("done", False))
    return {
        "phase": phase,
        "work_done": work_done,
        "work_total": work_total,
        "completed_files": completed_files,
        "remaining_files": remaining_files,
        "total_files": total_files,
        "analysis_state": analysis_state,
        "classification": classification,
        "event_kind": event_kind,
        "event_seq": event_seq,
        "stale_for_s": stale_for_s,
        "phase_progress_v2": normalized_phase_progress_v2,
        "progress_marker": progress_marker,
        "phase_timeline_header": (
            phase_timeline_header
            if isinstance(phase_timeline_header, str)
            else ""
        ),
        "phase_timeline_row": (
            phase_timeline_row if isinstance(phase_timeline_row, str) else ""
        ),
        "resume_checkpoint": normalized_resume_checkpoint,
        "done": done,
    }


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


def _emit_checkpoint_intro_timeline_progress(*, header: str | None, row: str) -> None:
    if isinstance(header, str) and header:
        typer.echo(header)
    typer.echo(row)


def _emit_analysis_resume_summary(result: JSONObject) -> None:
    resume = result.get("analysis_resume")
    if not isinstance(resume, Mapping):
        return
    path = str(resume.get("checkpoint_path", "") or "")
    status = str(resume.get("status", "") or "")
    reused_files = int(resume.get("reused_files", 0) or 0)
    total_files = int(resume.get("total_files", 0) or 0)
    remaining_files = int(resume.get("remaining_files", 0) or 0)
    cache_verdict = str(resume.get("cache_verdict", "") or "")
    status_suffix = f" status={status}" if status else ""
    verdict_suffix = f" cache_verdict={cache_verdict}" if cache_verdict else ""
    typer.echo(
        "Resume checkpoint: "
        f"path={path or '<none>'} reused_files={reused_files}/{total_files} "
        f"remaining_files={remaining_files}{status_suffix}{verdict_suffix}"
    )


def _context_run_dataflow_raw_argv(ctx: typer.Context) -> Callable[[list[str]], None]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_dataflow_raw_argv")
        if callable(candidate):
            return candidate
    return _run_dataflow_raw_argv


def _context_run_check(ctx: typer.Context) -> Callable[..., JSONObject]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_check")
        if callable(candidate):
            return candidate
    return run_check


def _context_run_with_timeout_retries(ctx: typer.Context) -> Callable[..., JSONObject]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_with_timeout_retries")
        if callable(candidate):
            return candidate
    return _run_with_timeout_retries


def _run_dataflow_raw_argv(
    argv: list[str],
    *,
    runner: Runner | None = None,
) -> None:
    opts = parse_dataflow_args_or_exit(argv)
    payload = build_dataflow_payload(opts)
    resolved_runner = runner or run_command
    startup_resume_emitted = False
    timeline_header_emitted = False
    last_phase_progress_signature: tuple[object, ...] | None = None
    last_phase_event_seq: int | None = None
    if opts.resume_checkpoint is not None:
        _emit_resume_checkpoint_startup_line(
            checkpoint_path=str(opts.resume_checkpoint),
            status="pending",
            reused_files=None,
            total_files=None,
        )

    def _on_notification(notification: JSONObject) -> None:
        nonlocal startup_resume_emitted
        nonlocal timeline_header_emitted
        nonlocal last_phase_progress_signature
        nonlocal last_phase_event_seq
        resume = _resume_checkpoint_from_progress_notification(notification)
        if not isinstance(resume, Mapping):
            pass
        elif not startup_resume_emitted:
            _emit_resume_checkpoint_startup_line(
                checkpoint_path=str(resume.get("checkpoint_path", "") or ""),
                status=str(resume.get("status", "") or ""),
                reused_files=int(resume.get("reused_files", 0) or 0),
                total_files=int(resume.get("total_files", 0) or 0),
            )
            startup_resume_emitted = True
        timeline_update = _phase_timeline_from_progress_notification(
            notification
        )
        phase_progress = _phase_progress_from_progress_notification(notification)
        if not isinstance(phase_progress, Mapping):
            return
        event_seq = phase_progress.get("event_seq")
        if isinstance(event_seq, int):
            if last_phase_event_seq == event_seq:
                return
            last_phase_event_seq = event_seq
        signature = (
            phase_progress.get("phase"),
            phase_progress.get("analysis_state"),
            phase_progress.get("classification"),
            phase_progress.get("event_kind"),
            phase_progress.get("event_seq"),
            phase_progress.get("work_done"),
            phase_progress.get("work_total"),
            phase_progress.get("completed_files"),
            phase_progress.get("remaining_files"),
            phase_progress.get("total_files"),
            phase_progress.get("stale_for_s"),
            phase_progress.get("progress_marker"),
            phase_progress.get("done"),
        )
        if signature == last_phase_progress_signature:
            return
        last_phase_progress_signature = signature
        if isinstance(timeline_update, Mapping):
            row = str(timeline_update.get("row") or "")
            header_value = timeline_update.get("header")
            header = (
                header_value
                if not timeline_header_emitted and isinstance(header_value, str) and header_value
                else None
            )
            _emit_checkpoint_intro_timeline_progress(header=header, row=row)
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
        emit_timeout_progress_report=opts.emit_timeout_progress_report,
        resume_on_timeout=max(int(opts.resume_on_timeout), 0),
    )
    _emit_dataflow_result_outputs(result, opts)
    _emit_analysis_resume_summary(result)
    _emit_nonzero_exit_causes(result)
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def check(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(None),
    profile: str = typer.Option("strict", "--profile"),
    report: Optional[Path] = typer.Option(None, "--report"),
    fail_on_violations: bool = typer.Option(True, "--fail-on-violations/--no-fail-on-violations"),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    decision_snapshot: Optional[Path] = typer.Option(
        None, "--decision-snapshot", help="Write decision surface snapshot JSON."
    ),
    emit_test_obsolescence: bool = typer.Option(
        False,
        "--emit-test-obsolescence/--no-emit-test-obsolescence",
        help=(
            "Write test obsolescence report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_obsolescence_state: bool = typer.Option(
        False,
        "--emit-test-obsolescence-state/--no-emit-test-obsolescence-state",
        help="Write test obsolescence state to artifacts/out.",
    ),
    test_obsolescence_state: Optional[Path] = typer.Option(
        None,
        "--test-obsolescence-state",
        help="Use precomputed test obsolescence state for delta/report.",
    ),
    emit_test_obsolescence_delta: bool = typer.Option(
        False,
        "--emit-test-obsolescence-delta/--no-emit-test-obsolescence-delta",
        help=(
            "Write test obsolescence delta report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_evidence_suggestions: bool = typer.Option(
        False,
        "--emit-test-evidence-suggestions/--no-emit-test-evidence-suggestions",
        help=(
            "Write test evidence suggestions (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_call_clusters: bool = typer.Option(
        False,
        "--emit-call-clusters/--no-emit-call-clusters",
        help="Write call cluster report (JSON in artifacts/out, markdown in out/).",
    ),
    emit_call_cluster_consolidation: bool = typer.Option(
        False,
        "--emit-call-cluster-consolidation/--no-emit-call-cluster-consolidation",
        help=(
            "Write call cluster consolidation plan (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_test_annotation_drift: bool = typer.Option(
        False,
        "--emit-test-annotation-drift/--no-emit-test-annotation-drift",
        help=(
            "Write test annotation drift report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    emit_semantic_coverage_map: bool = typer.Option(
        False,
        "--emit-semantic-coverage-map/--no-emit-semantic-coverage-map",
        help=(
            "Write semantic coverage map report (JSON in artifacts/out, markdown in artifacts/audit_reports/)."
        ),
    ),
    semantic_coverage_mapping: Optional[Path] = typer.Option(
        None,
        "--semantic-coverage-mapping",
        help="Use explicit semantic coverage mapping JSON (defaults to out/semantic_coverage_mapping.json).",
    ),
    test_annotation_drift_state: Optional[Path] = typer.Option(
        None,
        "--test-annotation-drift-state",
        help="Use precomputed annotation drift state for delta.",
    ),
    emit_test_annotation_drift_delta: bool = typer.Option(
        False,
        "--emit-test-annotation-drift-delta/--no-emit-test-annotation-drift-delta",
        help=(
            "Write test annotation drift delta report (JSON in artifacts/out, markdown in out/)."
        ),
    ),
    write_test_annotation_drift_baseline: bool = typer.Option(
        False,
        "--write-test-annotation-drift-baseline/--no-write-test-annotation-drift-baseline",
        help="Write the current test annotation drift baseline to baselines/.",
    ),
    write_test_obsolescence_baseline: bool = typer.Option(
        False,
        "--write-test-obsolescence-baseline/--no-write-test-obsolescence-baseline",
        help="Write the current test obsolescence baseline to baselines/.",
    ),
    emit_ambiguity_delta: bool = typer.Option(
        False,
        "--emit-ambiguity-delta/--no-emit-ambiguity-delta",
        help="Write ambiguity delta report (JSON in artifacts/out, markdown in out/).",
    ),
    emit_ambiguity_state: bool = typer.Option(
        False,
        "--emit-ambiguity-state/--no-emit-ambiguity-state",
        help="Write ambiguity state to artifacts/out.",
    ),
    ambiguity_state: Optional[Path] = typer.Option(
        None,
        "--ambiguity-state",
        help="Use precomputed ambiguity state for delta.",
    ),
    write_ambiguity_baseline: bool = typer.Option(
        False,
        "--write-ambiguity-baseline/--no-write-ambiguity-baseline",
        help="Write the current ambiguity baseline to baselines/.",
    ),
    baseline: Optional[Path] = typer.Option(
        None, "--baseline", help="Baseline file of allowed violations."
    ),
    baseline_write: bool = typer.Option(
        False, "--baseline-write", help="Write current violations to baseline."
    ),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params_csv: Optional[str] = typer.Option(None, "--ignore-params"),
    transparent_decorators_csv: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    resume_checkpoint: Optional[Path] = typer.Option(
        None,
        "--resume-checkpoint",
        help="Checkpoint path for resumable timeout runs.",
    ),
    emit_timeout_progress_report: bool = typer.Option(
        False,
        "--emit-timeout-progress-report/--no-emit-timeout-progress-report",
        help="Write timeout progress report artifacts on timeout.",
    ),
    resume_on_timeout: int = typer.Option(
        0,
        "--resume-on-timeout",
        min=0,
        help="Retry count when timeout reports timed_out_progress_resume.",
    ),
    analysis_tick_limit: Optional[int] = typer.Option(
        None,
        "--analysis-tick-limit",
        min=1,
        help="Deterministic logical timeout budget (ticks).",
    ),
    fail_on_type_ambiguities: bool = typer.Option(
        True, "--fail-on-type-ambiguities/--no-fail-on-type-ambiguities"
    ),
    lint: bool = typer.Option(False, "--lint/--no-lint"),
    lint_jsonl: Optional[Path] = typer.Option(
        None, "--lint-jsonl", help="Write lint JSONL to file or '-' for stdout."
    ),
    lint_sarif: Optional[Path] = typer.Option(
        None, "--lint-sarif", help="Write lint SARIF to file or '-' for stdout."
    ),
) -> None:
    # dataflow-bundle: filter_bundle
    """Run the dataflow grammar audit with strict defaults."""
    profile_name = profile.strip().lower()
    filter_bundle = DataflowFilterBundle(
        ignore_params_csv=ignore_params_csv,
        transparent_decorators_csv=transparent_decorators_csv,
    )
    if profile_name not in {"strict", "raw"}:
        raise typer.BadParameter("profile must be 'strict' or 'raw'")
    if profile_name == "raw":
        run_dataflow_raw_argv_fn = _context_run_dataflow_raw_argv(ctx)
        _run_check_raw_profile(
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
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
            fail_on_type_ambiguities=fail_on_type_ambiguities,
            lint=lint,
            lint_jsonl=lint_jsonl,
            lint_sarif=lint_sarif,
            run_dataflow_raw_argv_fn=run_dataflow_raw_argv_fn,
        )
        return
    extra_tokens = list(ctx.args)
    extra_tokens.extend(
        str(path) for path in (paths or []) if str(path).startswith("-")
    )
    if extra_tokens:
        joined = " ".join(extra_tokens)
        raise typer.BadParameter(
            f"Unknown arguments for strict profile: {joined}. Use --profile raw for raw options."
        )
    lint_enabled = lint or bool(lint_jsonl or lint_sarif)
    startup_resume_emitted = False
    timeline_header_emitted = False
    last_phase_progress_signature: tuple[object, ...] | None = None
    last_phase_event_seq: int | None = None
    if resume_checkpoint is not None:
        _emit_resume_checkpoint_startup_line(
            checkpoint_path=str(resume_checkpoint),
            status="pending",
            reused_files=None,
            total_files=None,
        )

    def _on_notification(notification: JSONObject) -> None:
        nonlocal startup_resume_emitted
        nonlocal timeline_header_emitted
        nonlocal last_phase_progress_signature
        nonlocal last_phase_event_seq
        resume = _resume_checkpoint_from_progress_notification(notification)
        if not isinstance(resume, Mapping):
            pass
        elif not startup_resume_emitted:
            _emit_resume_checkpoint_startup_line(
                checkpoint_path=str(resume.get("checkpoint_path", "") or ""),
                status=str(resume.get("status", "") or ""),
                reused_files=int(resume.get("reused_files", 0) or 0),
                total_files=int(resume.get("total_files", 0) or 0),
            )
            startup_resume_emitted = True
        timeline_update = _phase_timeline_from_progress_notification(
            notification
        )
        phase_progress = _phase_progress_from_progress_notification(notification)
        if not isinstance(phase_progress, Mapping):
            return
        event_seq = phase_progress.get("event_seq")
        if isinstance(event_seq, int):
            if last_phase_event_seq == event_seq:
                return
            last_phase_event_seq = event_seq
        signature = (
            phase_progress.get("phase"),
            phase_progress.get("analysis_state"),
            phase_progress.get("classification"),
            phase_progress.get("event_kind"),
            phase_progress.get("event_seq"),
            phase_progress.get("work_done"),
            phase_progress.get("work_total"),
            phase_progress.get("completed_files"),
            phase_progress.get("remaining_files"),
            phase_progress.get("total_files"),
            phase_progress.get("stale_for_s"),
            phase_progress.get("progress_marker"),
            phase_progress.get("done"),
        )
        if signature == last_phase_progress_signature:
            return
        last_phase_progress_signature = signature
        if isinstance(timeline_update, Mapping):
            row = str(timeline_update.get("row") or "")
            header_value = timeline_update.get("header")
            header = (
                header_value
                if not timeline_header_emitted and isinstance(header_value, str) and header_value
                else None
            )
            _emit_checkpoint_intro_timeline_progress(header=header, row=row)
            if header is not None:
                timeline_header_emitted = True

    run_check_fn = _context_run_check(ctx)
    run_with_timeout_retries_fn = _context_run_with_timeout_retries(ctx)
    result = run_with_timeout_retries_fn(
        run_once=lambda: run_check_fn(
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
            artifact_flags=CheckArtifactFlags(
                emit_test_obsolescence=emit_test_obsolescence,
                emit_test_evidence_suggestions=emit_test_evidence_suggestions,
                emit_call_clusters=emit_call_clusters,
                emit_call_cluster_consolidation=emit_call_cluster_consolidation,
                emit_test_annotation_drift=emit_test_annotation_drift,
                emit_semantic_coverage_map=emit_semantic_coverage_map,
            ),
            delta_options=CheckDeltaOptions(
                emit_test_obsolescence_state=emit_test_obsolescence_state,
                test_obsolescence_state=test_obsolescence_state,
                emit_test_obsolescence_delta=emit_test_obsolescence_delta,
                test_annotation_drift_state=test_annotation_drift_state,
                emit_test_annotation_drift_delta=emit_test_annotation_drift_delta,
                write_test_annotation_drift_baseline=write_test_annotation_drift_baseline,
                write_test_obsolescence_baseline=write_test_obsolescence_baseline,
                semantic_coverage_mapping=semantic_coverage_mapping,
                emit_ambiguity_delta=emit_ambiguity_delta,
                emit_ambiguity_state=emit_ambiguity_state,
                ambiguity_state=ambiguity_state,
                write_ambiguity_baseline=write_ambiguity_baseline,
            ),
            exclude=exclude,
            filter_bundle=filter_bundle,
            allow_external=allow_external,
            strictness=strictness,
            resume_checkpoint=resume_checkpoint,
            emit_timeout_progress_report=emit_timeout_progress_report,
            resume_on_timeout=resume_on_timeout,
            analysis_tick_limit=analysis_tick_limit,
            notification_callback=_on_notification,
        ),
        root=Path(root),
        emit_timeout_progress_report=emit_timeout_progress_report,
        resume_on_timeout=resume_on_timeout,
    )
    with _cli_deadline_scope():
        lint_lines = result.get("lint_lines", []) or []
        _emit_lint_outputs(
            lint_lines,
            lint=lint,
            lint_jsonl=lint_jsonl,
            lint_sarif=lint_sarif,
        )
    _emit_analysis_resume_summary(result)
    _emit_nonzero_exit_causes(result)
    raise typer.Exit(code=int(result.get("exit_code", 0)))


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
        "--resume-checkpoint",
        default=None,
        help="Checkpoint path for resumable timeout runs.",
    )
    parser.add_argument(
        "--emit-timeout-progress-report",
        action="store_true",
        help="Write timeout progress report artifacts on timeout.",
    )
    parser.add_argument(
        "--resume-on-timeout",
        type=int,
        default=0,
        help="Retry count when timeout returns timed_out_progress_resume.",
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


def _load_audit_tools_module(
    *,
    audit_tools_path: Path | None = None,
    spec_from_file_location_fn: Callable[..., object | None] = importlib.util.spec_from_file_location,
    module_from_spec_fn: Callable[..., object] = importlib.util.module_from_spec,
    sys_path_list: list[str] | None = None,
    sys_modules_map: MutableMapping[str, object] | None = None,
) -> Generator[object, None, None]:
    repo_root = _find_repo_root()
    module_path = audit_tools_path or (repo_root / "scripts" / "audit_tools.py")
    if not module_path.exists():
        raise FileNotFoundError("audit_tools.py not found; repository layout required")

    @contextmanager
    def _load_audit_tools() -> Generator[object, None, None]:
        module_name = "gabion_repo_audit_tools"
        spec = spec_from_file_location_fn(module_name, module_path)
        loader = getattr(spec, "loader", None)
        if spec is None or loader is None:
            raise RuntimeError("failed to load audit_tools module")
        module = module_from_spec_fn(spec)
        scripts_root = str(module_path.parent)
        path_list = sys_path_list if sys_path_list is not None else sys.path
        modules = sys_modules_map if sys_modules_map is not None else sys.modules
        inserted_path = False
        if scripts_root not in path_list:
            path_list.insert(0, scripts_root)
            inserted_path = True
        previous_module = modules.get(module_name)
        had_previous = module_name in modules
        modules[module_name] = module
        try:
            loader.exec_module(module)
            yield module
        finally:
            if had_previous:
                modules[module_name] = previous_module  # type: ignore[assignment]
            else:
                modules.pop(module_name, None)
            if inserted_path:
                try:
                    path_list.remove(scripts_root)
                except ValueError:
                    pass

    return _load_audit_tools()


def _run_governance_cli(
    *,
    runner_name: str,
    args: list[str],
    audit_tools_path: Path | None = None,
    spec_from_file_location_fn: Callable[..., object | None] = importlib.util.spec_from_file_location,
    module_from_spec_fn: Callable[..., object] = importlib.util.module_from_spec,
    sys_path_list: list[str] | None = None,
    sys_modules_map: MutableMapping[str, object] | None = None,
) -> int:
    try:
        loader = _load_audit_tools_module(
            audit_tools_path=audit_tools_path,
            spec_from_file_location_fn=spec_from_file_location_fn,
            module_from_spec_fn=module_from_spec_fn,
            sys_path_list=sys_path_list,
            sys_modules_map=sys_modules_map,
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        return 2
    except Exception as exc:
        typer.secho(
            f"failed to load audit_tools module: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 2

    try:
        with loader as module:
            runner = getattr(module, runner_name, None)
            if runner is None:
                typer.secho(
                    f"audit_tools missing runner: {runner_name}",
                    err=True,
                    fg=typer.colors.RED,
                )
                return 2
            return int(runner(args))
    except Exception as exc:
        typer.secho(
            f"governance command failed ({runner_name}): {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 1


def _restore_dataflow_resume_checkpoint_from_github_artifacts(
    *,
    token: str,
    repo: str,
    output_dir: Path,
    ref_name: str = "",
    current_run_id: str = "",
    artifact_name: str = "dataflow-report",
    checkpoint_name: str = "dataflow_resume_checkpoint_ci.json",
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
    checkpoint_name = checkpoint_name.strip() or "dataflow_resume_checkpoint_ci.json"
    if not token or not repo:
        typer.echo("GitHub token/repository unavailable; skipping checkpoint restore.")
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
        typer.echo(f"Unable to query prior artifacts ({exc}); skipping checkpoint restore.")
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
    chunk_prefix = f"{checkpoint_name}.chunks/"
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
                    if base == checkpoint_name:
                        checkpoint_member = name
                    elif base.startswith(chunk_prefix):
                        chunk_members.append(name)
                if checkpoint_member is None:
                    continue
                checkpoint_bytes = zf.read(checkpoint_member)
                if _checkpoint_requires_chunk_artifacts(
                    checkpoint_bytes=checkpoint_bytes
                ) and not chunk_members:
                    continue
                checkpoint_output = output_dir / checkpoint_name
                chunk_output_dir = output_dir / f"{checkpoint_name}.chunks"
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
                    f"Restored {restored} checkpoint artifact file(s) from prior run."
                )
                return 0
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        typer.echo(
            f"Unable to restore checkpoint from prior artifacts ({last_error}); continuing without checkpoint."
        )
        return 0
    typer.echo(
        "Prior artifacts did not include usable resume checkpoint files; continuing without restore."
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


def _checkpoint_requires_chunk_artifacts(*, checkpoint_bytes: bytes) -> bool:
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


def _context_restore_resume_checkpoint(
    ctx: typer.Context,
) -> Callable[..., int]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("restore_resume_checkpoint")
        if callable(candidate):
            return candidate
    return _restore_dataflow_resume_checkpoint_from_github_artifacts


@app.command("restore-resume-checkpoint")
def restore_resume_checkpoint(
    ctx: typer.Context,
    token: str = typer.Option("", "--token", envvar="GH_TOKEN"),
    repo: str = typer.Option("", "--repo", envvar="GH_REPO"),
    output_dir: Path = typer.Option(Path("artifacts/audit_reports"), "--output-dir"),
    ref_name: str = typer.Option("", "--ref-name", envvar="GH_REF_NAME"),
    run_id: str = typer.Option("", "--run-id", envvar="GH_RUN_ID"),
    artifact_name: str = typer.Option("dataflow-report", "--artifact-name"),
    checkpoint_name: str = typer.Option(
        "dataflow_resume_checkpoint_ci.json", "--checkpoint-name"
    ),
) -> None:
    """Restore dataflow resume checkpoint files from prior workflow artifacts."""
    restore_fn = _context_restore_resume_checkpoint(ctx)
    exit_code = restore_fn(
        token=token,
        repo=repo,
        output_dir=output_dir,
        ref_name=ref_name,
        current_run_id=run_id,
        artifact_name=artifact_name,
        checkpoint_name=checkpoint_name,
    )
    raise typer.Exit(code=exit_code)


def _run_docflow_audit(
    *,
    root: Path,
    fail_on_violations: bool,
    sppf_gh_ref_mode: str = "required",
    extra_path: list[str] | None = None,
    audit_tools_path: Path | None = None,
    spec_from_file_location_fn: Callable[..., object | None] = importlib.util.spec_from_file_location,
    module_from_spec_fn: Callable[..., object] = importlib.util.module_from_spec,
    sys_path_list: list[str] | None = None,
    sys_modules_map: MutableMapping[str, object] | None = None,
) -> int:
    try:
        loader = _load_audit_tools_module(
            audit_tools_path=audit_tools_path,
            spec_from_file_location_fn=spec_from_file_location_fn,
            module_from_spec_fn=module_from_spec_fn,
            sys_path_list=sys_path_list,
            sys_modules_map=sys_modules_map,
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        return 2
    except Exception as exc:
        typer.secho(
            f"failed to load audit_tools module: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 2

    try:
        with loader as module:
            args = ["--root", str(root)]
            for entry in extra_path or []:
                args.extend(["--extra-path", entry])
            if fail_on_violations:
                args.append("--fail-on-violations")
            args.extend(["--sppf-gh-ref-mode", sppf_gh_ref_mode])
            status = int(module.run_docflow_cli(args))
            if status == 0:
                try:
                    status = int(module.run_sppf_graph_cli([]))
                except Exception as exc:
                    typer.secho(
                        f"docflow: sppf-graph failed: {exc}",
                        err=True,
                        fg=typer.colors.RED,
                    )
                    return 1
            return status
    except Exception as exc:
        typer.secho(
            f"failed to load audit_tools module: {exc}",
            err=True,
            fg=typer.colors.RED,
        )
        return 2


def _context_run_sppf_sync(ctx: typer.Context) -> Callable[..., int]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_sppf_sync")
        if callable(candidate):
            return candidate
    return _run_sppf_sync


def _context_run_governance_cli(ctx: typer.Context) -> Callable[..., int]:
    obj = ctx.obj
    if isinstance(obj, Mapping):
        candidate = obj.get("run_governance_cli")
        if callable(candidate):
            return candidate
    return _run_governance_cli


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
    run_sppf_sync_fn = _context_run_sppf_sync(ctx)
    with _cli_deadline_scope():
        try:
            exit_code = run_sppf_sync_fn(
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
    ctx: typer.Context,
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
    run_governance_cli_fn = _context_run_governance_cli(ctx)
    exit_code = run_governance_cli_fn(runner_name="run_sppf_graph_cli", args=args)
    raise typer.Exit(code=exit_code)


@app.command("status-consistency")
def status_consistency(
    ctx: typer.Context,
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
    run_governance_cli_fn = _context_run_governance_cli(ctx)
    exit_code = run_governance_cli_fn(runner_name="run_status_consistency_cli", args=args)
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
    output = json.dumps(normalized, indent=2, sort_keys=True)
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
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
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
    typer.echo(json.dumps(normalized, indent=2, sort_keys=True))
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
    typer.echo(json.dumps(normalized, indent=2, sort_keys=True))
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
    typer.echo(json.dumps(normalized, indent=2, sort_keys=True))
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
    output = json.dumps(normalized, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        _write_text_to_target(output_path, output)
