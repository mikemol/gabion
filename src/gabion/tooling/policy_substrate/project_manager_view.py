from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from gabion.analysis.semantics.report_doc import ReportDoc
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_graph import load_invariant_workstreams
from gabion.tooling.policy_substrate.view_dsl import (
    AddIntExpr,
    CoalesceExpr,
    PathExpr,
    WeightedSumExpr,
    WeightedTerm,
    collection_items,
    eval_int,
    eval_mapping,
    eval_text,
)
from gabion.tooling.runtime.declarative_script_host import (
    DeclarativeScriptSpec,
    ScriptInvocation,
    ScriptOptionKind,
    ScriptOptionSpec,
    invoke_script,
    script_runtime_scope,
)

_FORMAT_VERSION = 1
_DEFAULT_SOURCE_ARTIFACT = "artifacts/out/invariant_workstreams.json"
_DEFAULT_OUT = "artifacts/out/project_manager_view.json"
_DEFAULT_MARKDOWN_OUT = "artifacts/out/project_manager_view.md"
_DEFAULT_MERMAID_OUT = "artifacts/out/project_manager_view.mmd"
_DEFAULT_VISUAL_LIMIT = 6


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="gabion.tooling.policy_substrate.project_manager_view",
        key=key,
    )


def _temp_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.tmp")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path(path)
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _status_count_payload(workstreams: tuple["ProjectManagerWorkstream", ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for workstream in workstreams:
        counts[workstream.status] = counts.get(workstream.status, 0) + 1
    return {
        key: counts[key]
        for key in _sorted(list(counts), key=lambda item: item)
    }


def _mermaid_label(*parts: str) -> str:
    escaped = [
        part.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")
        for part in parts
        if part
    ]
    return "<br/>".join(escaped)


@dataclass(frozen=True)
class ProjectManagerAction:
    followup_family: str
    action_kind: str
    object_id: str
    owner_root_object_id: str
    title: str
    blocker_class: str
    recommended_action: str

    def as_payload(self) -> dict[str, object]:
        return {
            "followup_family": self.followup_family,
            "action_kind": self.action_kind,
            "object_id": self.object_id,
            "owner_root_object_id": self.owner_root_object_id,
            "title": self.title,
            "blocker_class": self.blocker_class,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class ProjectManagerWorkstream:
    object_id: str
    title: str
    status: str
    touchsite_count: int
    surviving_touchsite_count: int
    policy_signal_count: int
    diagnostic_count: int
    failing_test_case_count: int
    test_failure_count: int
    coverage_count: int
    doc_alignment_pressure: int
    pressure_score: int
    recommended_followup: ProjectManagerAction | None

    def as_payload(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "title": self.title,
            "status": self.status,
            "touchsite_count": self.touchsite_count,
            "surviving_touchsite_count": self.surviving_touchsite_count,
            "policy_signal_count": self.policy_signal_count,
            "diagnostic_count": self.diagnostic_count,
            "failing_test_case_count": self.failing_test_case_count,
            "test_failure_count": self.test_failure_count,
            "coverage_count": self.coverage_count,
            "doc_alignment_pressure": self.doc_alignment_pressure,
            "pressure_score": self.pressure_score,
            "recommended_followup": (
                None
                if self.recommended_followup is None
                else self.recommended_followup.as_payload()
            ),
        }


@dataclass(frozen=True)
class ProjectManagerPortfolioSummary:
    workstream_count: int
    status_counts: Mapping[str, int]
    total_touchsites: int
    total_surviving_touchsites: int
    total_policy_signals: int
    total_diagnostics: int
    total_failing_test_cases: int
    total_test_failures: int
    total_coverage_hits: int
    total_doc_alignment_pressure: int
    dominant_followup_class: str
    next_human_followup_family: str

    def as_payload(self) -> dict[str, object]:
        return {
            "workstream_count": self.workstream_count,
            "status_counts": dict(self.status_counts),
            "total_touchsites": self.total_touchsites,
            "total_surviving_touchsites": self.total_surviving_touchsites,
            "total_policy_signals": self.total_policy_signals,
            "total_diagnostics": self.total_diagnostics,
            "total_failing_test_cases": self.total_failing_test_cases,
            "total_test_failures": self.total_test_failures,
            "total_coverage_hits": self.total_coverage_hits,
            "total_doc_alignment_pressure": self.total_doc_alignment_pressure,
            "dominant_followup_class": self.dominant_followup_class,
            "next_human_followup_family": self.next_human_followup_family,
        }


@dataclass(frozen=True)
class ProjectManagerView:
    source_artifact: str
    source_generated_at_utc: str
    generated_at_utc: str
    visual_limit: int
    portfolio_summary: ProjectManagerPortfolioSummary
    repo_next_action: ProjectManagerAction | None
    repo_code_followup: ProjectManagerAction | None
    workstreams: tuple[ProjectManagerWorkstream, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "source_artifact": self.source_artifact,
            "source_generated_at_utc": self.source_generated_at_utc,
            "generated_at_utc": self.generated_at_utc,
            "visual_limit": self.visual_limit,
            "portfolio_summary": self.portfolio_summary.as_payload(),
            "repo_next_action": (
                None if self.repo_next_action is None else self.repo_next_action.as_payload()
            ),
            "repo_code_followup": (
                None
                if self.repo_code_followup is None
                else self.repo_code_followup.as_payload()
            ),
            "workstreams": [item.as_payload() for item in self.workstreams],
            "visualization_mermaid": render_mermaid(self),
        }


def _action_from_payload(payload: Mapping[str, object]) -> ProjectManagerAction | None:
    title = eval_text(PathExpr("title"), payload)
    followup_family = eval_text(PathExpr("followup_family"), payload)
    action_kind = eval_text(PathExpr("action_kind"), payload)
    object_id = eval_text(PathExpr("object_id"), payload)
    owner_root_object_id = eval_text(PathExpr("owner_root_object_id"), payload)
    blocker_class = eval_text(PathExpr("blocker_class"), payload)
    recommended_action = eval_text(PathExpr("recommended_action"), payload)
    if not any(
        (
            title,
            followup_family,
            action_kind,
            object_id,
            owner_root_object_id,
            blocker_class,
            recommended_action,
        )
    ):
        return None
    return ProjectManagerAction(
        followup_family=followup_family,
        action_kind=action_kind,
        object_id=object_id,
        owner_root_object_id=owner_root_object_id,
        title=title,
        blocker_class=blocker_class,
        recommended_action=recommended_action,
    )


_RECOMMENDED_FOLLOWUP_EXPR = CoalesceExpr(
    items=(
        PathExpr("recommended_followup"),
        PathExpr("next_actions.recommended_followup"),
    )
)
_DOC_ALIGNMENT_PRESSURE_EXPR = AddIntExpr(
    items=(
        PathExpr("doc_alignment_summary.missing_target_doc_count"),
        PathExpr("doc_alignment_summary.ambiguous_target_doc_count"),
        PathExpr("doc_alignment_summary.unassigned_target_doc_count"),
        PathExpr("doc_alignment_summary.append_pending_existing_target_doc_count"),
        PathExpr("doc_alignment_summary.append_pending_new_target_doc_count"),
    )
)
_PRESSURE_SCORE_EXPR = WeightedSumExpr(
    items=(
        WeightedTerm(weight=3, expr=PathExpr("surviving_touchsite_count")),
        WeightedTerm(weight=5, expr=PathExpr("policy_signal_count")),
        WeightedTerm(weight=8, expr=PathExpr("diagnostic_count")),
        WeightedTerm(weight=4, expr=PathExpr("failing_test_case_count")),
        WeightedTerm(weight=6, expr=PathExpr("test_failure_count")),
        WeightedTerm(weight=2, expr=_DOC_ALIGNMENT_PRESSURE_EXPR),
    )
)


def analyze(
    *,
    payload: Mapping[str, object],
    source_artifact: str,
    visual_limit: int = _DEFAULT_VISUAL_LIMIT,
) -> ProjectManagerView:
    workstreams: list[ProjectManagerWorkstream] = []
    for workstream_payload in collection_items(payload, "workstreams"):
        doc_alignment_pressure = eval_int(_DOC_ALIGNMENT_PRESSURE_EXPR, workstream_payload)
        recommended_followup = _action_from_payload(
            eval_mapping(_RECOMMENDED_FOLLOWUP_EXPR, workstream_payload)
        )
        workstream = ProjectManagerWorkstream(
            object_id=eval_text(PathExpr("object_id"), workstream_payload),
            title=eval_text(PathExpr("title"), workstream_payload),
            status=eval_text(PathExpr("status"), workstream_payload) or "unknown",
            touchsite_count=eval_int(PathExpr("touchsite_count"), workstream_payload),
            surviving_touchsite_count=eval_int(
                PathExpr("surviving_touchsite_count"),
                workstream_payload,
            ),
            policy_signal_count=eval_int(
                PathExpr("policy_signal_count"),
                workstream_payload,
            ),
            diagnostic_count=eval_int(PathExpr("diagnostic_count"), workstream_payload),
            failing_test_case_count=eval_int(
                PathExpr("failing_test_case_count"),
                workstream_payload,
            ),
            test_failure_count=eval_int(
                PathExpr("test_failure_count"),
                workstream_payload,
            ),
            coverage_count=eval_int(PathExpr("coverage_count"), workstream_payload),
            doc_alignment_pressure=doc_alignment_pressure,
            pressure_score=eval_int(_PRESSURE_SCORE_EXPR, workstream_payload),
            recommended_followup=recommended_followup,
        )
        workstreams.append(workstream)
    ordered_workstreams = tuple(
        _sorted(
            workstreams,
            key=lambda item: (
                -item.pressure_score,
                -item.surviving_touchsite_count,
                item.object_id,
            ),
        )
    )
    repo_next_actions = eval_mapping(PathExpr("repo_next_actions"), payload)
    portfolio_summary = ProjectManagerPortfolioSummary(
        workstream_count=len(ordered_workstreams),
        status_counts=_status_count_payload(ordered_workstreams),
        total_touchsites=sum(item.touchsite_count for item in ordered_workstreams),
        total_surviving_touchsites=sum(
            item.surviving_touchsite_count for item in ordered_workstreams
        ),
        total_policy_signals=sum(item.policy_signal_count for item in ordered_workstreams),
        total_diagnostics=sum(item.diagnostic_count for item in ordered_workstreams),
        total_failing_test_cases=sum(
            item.failing_test_case_count for item in ordered_workstreams
        ),
        total_test_failures=sum(item.test_failure_count for item in ordered_workstreams),
        total_coverage_hits=sum(item.coverage_count for item in ordered_workstreams),
        total_doc_alignment_pressure=sum(
            item.doc_alignment_pressure for item in ordered_workstreams
        ),
        dominant_followup_class=eval_text(
            PathExpr("dominant_followup_class"),
            repo_next_actions,
        ),
        next_human_followup_family=eval_text(
            PathExpr("next_human_followup_family"),
            repo_next_actions,
        ),
    )
    return ProjectManagerView(
        source_artifact=source_artifact,
        source_generated_at_utc=eval_text(PathExpr("generated_at_utc"), payload),
        generated_at_utc=datetime.now(UTC).isoformat(),
        visual_limit=max(1, int(visual_limit)),
        portfolio_summary=portfolio_summary,
        repo_next_action=_action_from_payload(
            eval_mapping(PathExpr("recommended_followup"), repo_next_actions)
        ),
        repo_code_followup=_action_from_payload(
            eval_mapping(PathExpr("recommended_code_followup"), repo_next_actions)
        ),
        workstreams=ordered_workstreams,
    )


def render_mermaid(view: ProjectManagerView) -> str:
    lines = ["flowchart TB"]
    summary = view.portfolio_summary
    lines.append(
        '    pm["'
        + _mermaid_label(
            "Project Manager View",
            f"workstreams {summary.workstream_count}",
            f"surviving touchsites {summary.total_surviving_touchsites}",
            f"diagnostics {summary.total_diagnostics}",
            f"test failures {summary.total_test_failures}",
        )
        + '"]'
    )
    if view.repo_next_action is not None:
        lines.append(
            '    repo_next["'
            + _mermaid_label(
                "Repo Next Action",
                view.repo_next_action.object_id or view.repo_next_action.followup_family,
                view.repo_next_action.title,
            )
            + '"]'
        )
        lines.append("    pm --> repo_next")
    if (
        view.repo_code_followup is not None
        and view.repo_code_followup != view.repo_next_action
    ):
        lines.append(
            '    repo_code["'
            + _mermaid_label(
                "Repo Code Followup",
                view.repo_code_followup.object_id,
                view.repo_code_followup.title,
            )
            + '"]'
        )
        lines.append("    pm --> repo_code")
    for index, workstream in enumerate(view.workstreams[: view.visual_limit]):
        node_id = f"ws_{index}"
        lines.append(
            f'    {node_id}["'
            + _mermaid_label(
                workstream.object_id,
                workstream.status,
                f"pressure {workstream.pressure_score}",
                f"surviving {workstream.surviving_touchsite_count}",
            )
            + '"]'
        )
        lines.append(f"    pm --> {node_id}")
        if workstream.recommended_followup is not None:
            followup_id = f"fu_{index}"
            lines.append(
                f'    {followup_id}["'
                + _mermaid_label(
                    workstream.recommended_followup.object_id
                    or workstream.recommended_followup.followup_family,
                    workstream.recommended_followup.action_kind,
                    workstream.recommended_followup.title,
                )
                + '"]'
            )
            lines.append(f"    {node_id} --> {followup_id}")
            lines.append(f"    class {followup_id} action;")
        lines.append(f"    class {node_id} {workstream.status or 'unknown'};")
    lines.extend(
        (
            "    classDef in_progress fill:#f5d27a,stroke:#8d6400,color:#261800;",
            "    classDef landed fill:#bedfc4,stroke:#1f6b35,color:#11301b;",
            "    classDef queued fill:#cedbff,stroke:#365eb3,color:#16254f;",
            "    classDef blocked fill:#f4b2b0,stroke:#8d2f2c,color:#3b0c0a;",
            "    classDef unknown fill:#d9d9d9,stroke:#666666,color:#111111;",
            "    classDef action fill:#fff4d6,stroke:#8d6400,color:#342100;",
        )
    )
    return "\n".join(lines)


def render_markdown(view: ProjectManagerView) -> str:
    with script_runtime_scope():
        doc = ReportDoc("project_manager_view")
        summary = view.portfolio_summary
        doc.header(1, "Project Manager View")
        doc.bullets(
            (
                f"source_artifact: `{view.source_artifact}`",
                f"source_generated_at_utc: `{view.source_generated_at_utc}`",
                f"generated_at_utc: `{view.generated_at_utc}`",
            )
        )
        doc.header(2, "Portfolio Summary")
        doc.bullets(
            (
                f"workstream_count: `{summary.workstream_count}`",
                f"status_counts: `{json.dumps(dict(summary.status_counts), sort_keys=True)}`",
                f"surviving_touchsites: `{summary.total_surviving_touchsites}`",
                f"diagnostics: `{summary.total_diagnostics}`",
                f"test_failures: `{summary.total_test_failures}`",
                f"dominant_followup_class: `{summary.dominant_followup_class}`",
                f"next_human_followup_family: `{summary.next_human_followup_family}`",
            )
        )
        doc.header(2, "Recommended Actions")
        if view.repo_next_action is not None:
            doc.bullets(
                (
                    "repo_next_action:",
                    f"title: `{view.repo_next_action.title}`",
                    f"object_id: `{view.repo_next_action.object_id}`",
                    f"followup_family: `{view.repo_next_action.followup_family}`",
                    f"recommended_action: `{view.repo_next_action.recommended_action}`",
                )
            )
        if (
            view.repo_code_followup is not None
            and view.repo_code_followup != view.repo_next_action
        ):
            doc.bullets(
                (
                    "repo_code_followup:",
                    f"title: `{view.repo_code_followup.title}`",
                    f"object_id: `{view.repo_code_followup.object_id}`",
                    f"followup_family: `{view.repo_code_followup.followup_family}`",
                    f"recommended_action: `{view.repo_code_followup.recommended_action}`",
                )
            )
        doc.header(2, "Workstreams")
        doc.table(
            (
                "workstream_id",
                "status",
                "pressure",
                "surviving_touchsites",
                "policy_signals",
                "diagnostics",
                "test_failures",
                "doc_alignment",
                "recommended_followup",
            ),
            (
                (
                    item.object_id,
                    item.status,
                    item.pressure_score,
                    item.surviving_touchsite_count,
                    item.policy_signal_count,
                    item.diagnostic_count,
                    item.test_failure_count,
                    item.doc_alignment_pressure,
                    (
                        item.recommended_followup.object_id
                        if item.recommended_followup is not None
                        else ""
                    ),
                )
                for item in view.workstreams
            ),
        )
        doc.header(2, "Visualization")
        doc.codeblock(render_mermaid(view), language="mermaid")
        return doc.emit()


def run(
    *,
    source_artifact_path: Path,
    out_path: Path,
    markdown_out: Path,
    mermaid_out: Path,
    visual_limit: int = _DEFAULT_VISUAL_LIMIT,
) -> int:
    payload = load_invariant_workstreams(source_artifact_path)
    view = analyze(
        payload=payload,
        source_artifact=str(source_artifact_path),
        visual_limit=visual_limit,
    )
    _write_json(out_path, view.as_payload())
    _write_text(markdown_out, render_markdown(view))
    _write_text(mermaid_out, render_mermaid(view) + "\n")
    return 0


def _run_invocation(invocation: ScriptInvocation) -> int:
    return run(
        source_artifact_path=invocation.path("source_artifact"),
        out_path=invocation.path("out"),
        markdown_out=invocation.path("markdown_out"),
        mermaid_out=invocation.path("mermaid_out"),
        visual_limit=invocation.integer("visual_limit"),
    )


_SCRIPT_SPEC = DeclarativeScriptSpec(
    script_id="project_manager_view",
    description=__doc__ or "Render the project manager planning view.",
    options=(
        ScriptOptionSpec(
            dest="source_artifact",
            flags=("--source-artifact",),
            kind=ScriptOptionKind.PATH,
            default=_DEFAULT_SOURCE_ARTIFACT,
        ),
        ScriptOptionSpec(
            dest="out",
            flags=("--out",),
            kind=ScriptOptionKind.PATH,
            default=_DEFAULT_OUT,
        ),
        ScriptOptionSpec(
            dest="markdown_out",
            flags=("--markdown-out",),
            kind=ScriptOptionKind.PATH,
            default=_DEFAULT_MARKDOWN_OUT,
        ),
        ScriptOptionSpec(
            dest="mermaid_out",
            flags=("--mermaid-out",),
            kind=ScriptOptionKind.PATH,
            default=_DEFAULT_MERMAID_OUT,
        ),
        ScriptOptionSpec(
            dest="visual_limit",
            flags=("--visual-limit",),
            kind=ScriptOptionKind.INTEGER,
            default=_DEFAULT_VISUAL_LIMIT,
        ),
    ),
    handler=_run_invocation,
)


def main(argv: list[str] | None = None) -> int:
    return invoke_script(_SCRIPT_SPEC, argv=argv)


__all__ = [
    "ProjectManagerAction",
    "ProjectManagerPortfolioSummary",
    "ProjectManagerView",
    "ProjectManagerWorkstream",
    "analyze",
    "main",
    "render_markdown",
    "render_mermaid",
    "run",
]
