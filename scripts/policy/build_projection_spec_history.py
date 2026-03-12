#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from gabion.analysis.foundation.timeout_context import (
    Deadline,
    deadline_clock_scope,
    deadline_scope,
)
from gabion.analysis.projection.projection_registry import iter_registered_specs
from gabion.analysis.projection.projection_semantic_lowering import (
    BridgeProjectionOp,
    PresentationProjectionOp,
    ProjectionSemanticLoweringPlan,
    SemanticProjectionOp,
    lower_projection_spec_to_semantic_plan,
)
from gabion.deadline_clock import GasMeter
from gabion.order_contract import ordered_or_sorted

EraStatus = Literal["implemented", "in_progress", "open"]
CriterionStatus = Literal["pass", "fail", "in_progress"]

_MAX_COMMIT_EVIDENCE = 10
_MAX_TOP_HOTSPOTS = 10


@dataclass(frozen=True)
class CommitRecord:
    sha: str
    date: str
    subject: str


@dataclass(frozen=True)
class WorkspaceChange:
    status_code: str
    path: str


@dataclass(frozen=True)
class EraSpec:
    era_id: str
    title: str
    base_status: EraStatus
    intent_sources: tuple[str, ...]
    implementation_surfaces: tuple[str, ...]
    validation_surfaces: tuple[str, ...]
    completion_gaps: tuple[str, ...]
    next_actions: tuple[str, ...]


@dataclass(frozen=True)
class StatementPattern:
    path: str
    contains: tuple[str, ...]
    limit: int = 2


_ERA_SPECS: tuple[EraSpec, ...] = (
    EraSpec(
        era_id="PS-ERA-01",
        title="ProjectionSpec Core Calculus",
        base_status="implemented",
        intent_sources=(
            "in/in-30.md",
            "in/in-31.md",
        ),
        implementation_surfaces=(
            "src/gabion/analysis/projection/projection_spec.py",
            "src/gabion/analysis/projection/projection_normalize.py",
            "src/gabion/analysis/projection/projection_exec.py",
            "src/gabion/analysis/projection/projection_registry.py",
        ),
        validation_surfaces=(
            "tests/gabion/analysis/projection/test_projection_spec.py",
            "tests/gabion/analysis/projection/test_projection_exec_edges.py",
        ),
        completion_gaps=(),
        next_actions=(
            "Preserve deterministic normalization/hash behavior while convergence layers evolve.",
        ),
    ),
    EraSpec(
        era_id="PS-ERA-02",
        title="Quotient And Internment Formalization",
        base_status="implemented",
        intent_sources=(
            "in/in-31.md",
            "in/in-30.md",
        ),
        implementation_surfaces=(
            "src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py",
            "src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py",
        ),
        validation_surfaces=(
            "tests/gabion/analysis/projection/test_suite_order_projection_spec.py",
            "tests/gabion/analysis/dataflow_s1/test_dataflow_projection_helpers.py",
        ),
        completion_gaps=(),
        next_actions=(
            "Keep quotient/internment invariants tied to canonical projection owner paths.",
        ),
    ),
    EraSpec(
        era_id="PS-ERA-03",
        title="WS5 Dataflow Projection Ownerization",
        base_status="implemented",
        intent_sources=(
            "docs/ws5_decomposition_ledger.md",
            "docs/audits/dataflow_runtime_debt_ledger.md",
        ),
        implementation_surfaces=(
            "src/gabion/analysis/dataflow/engine/dataflow_projection_materialization.py",
            "src/gabion/analysis/dataflow/io/dataflow_projection_helpers.py",
            "src/gabion/analysis/dataflow/io/dataflow_reporting_helpers.py",
        ),
        validation_surfaces=(
            "tests/gabion/analysis/dataflow_s1/test_dataflow_report_helpers.py",
            "tests/gabion/analysis/projection/test_suite_order_projection_spec.py",
        ),
        completion_gaps=(),
        next_actions=(
            "Prevent compatibility wrapper drift by keeping projection/reporting helper ownership centralized.",
        ),
    ),
    EraSpec(
        era_id="PS-ERA-04",
        title="Policy DSL Convergence",
        base_status="implemented",
        intent_sources=(
            "docs/governance_control_loops.md",
            "docs/enforceable_rules_cheat_sheet.md",
            "docs/policy_dsl_migration_notes.md",
        ),
        implementation_surfaces=(
            "src/gabion/policy_dsl/schema.py",
            "src/gabion/policy_dsl/compile.py",
            "src/gabion/policy_dsl/typecheck.py",
            "src/gabion/policy_dsl/eval.py",
            "src/gabion/policy_dsl/registry.py",
            "src/gabion/analysis/aspf_rule_engine.py",
            "scripts/policy/policy_check.py",
        ),
        validation_surfaces=(
            "tests/test_policy_dsl.py",
        ),
        completion_gaps=(),
        next_actions=(
            "Preserve rule_id/witness drift checks so policy decisions stay DSL-owned.",
        ),
    ),
    EraSpec(
        era_id="PS-ERA-05",
        title="Fiber-First Lattice Cutover",
        base_status="implemented",
        intent_sources=(
            "docs/aspf_execution_fibration.md",
            "docs/projection_fiber_rules.yaml",
        ),
        implementation_surfaces=(
            "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
            "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
            "src/gabion/tooling/policy_rules/branchless_rule.py",
            "src/gabion/tooling/runtime/policy_scanner_suite.py",
        ),
        validation_surfaces=(
            "tests/test_policy_dsl.py",
            "tests/gabion/analysis/aspf/test_aspf_execution_fibration.py",
        ),
        completion_gaps=(),
        next_actions=(
            "Keep iterator-first convergence and single-frontier drift tests as required gates.",
        ),
    ),
    EraSpec(
        era_id="PS-ERA-06",
        title="Integrated Substrate Completion",
        base_status="in_progress",
        intent_sources=(
            "docs/projection_fiber_rules.yaml",
            "docs/policy_dsl_migration_notes.md",
            "docs/governance_loop_matrix.md",
            "in/in-32.md",
        ),
        implementation_surfaces=(
            "scripts/policy/policy_check.py",
            "src/gabion/tooling/policy_substrate/lattice_convergence_semantic.py",
            "src/gabion/tooling/policy_substrate/dataflow_fibration.py",
            "src/gabion/analysis/aspf/aspf_lattice_algebra.py",
        ),
        validation_surfaces=(
            "tests/test_policy_dsl.py",
            "scripts/policy/policy_check.py",
        ),
        completion_gaps=(
            "Workflow policy gate stack still reports unresolved workflow/lock-in failures.",
        ),
        next_actions=(
            "Close workflow-policy and lock-in source gate failures in a dedicated correction unit.",
            "Keep strict docflow packet loop green while CF04-CF11 substrate state remains stable.",
        ),
    ),
)


_STATEMENT_PATTERNS: tuple[StatementPattern, ...] = (
    StatementPattern(
        path="in/in-30.md",
        contains=("ProjectionSpec remains unchanged", "Projection Idempotence"),
    ),
    StatementPattern(
        path="in/in-31.md",
        contains=("ProjectionSpec is a quotient", "gauge-fixing inverse"),
    ),
    StatementPattern(
        path="docs/ws5_decomposition_ledger.md",
        contains=(
            "_materialize_projection_spec_rows",
            "_topologically_order_report_projection_specs",
            "ReportProjectionSpec",
        ),
        limit=4,
    ),
    StatementPattern(
        path="docs/audits/dataflow_runtime_debt_ledger.md",
        contains=("DFD-037", "DFD-038"),
    ),
    StatementPattern(
        path="docs/governance_control_loops.md",
        contains=("policy_dsl/compile.py", "policy_dsl/eval.py"),
    ),
    StatementPattern(
        path="docs/enforceable_rules_cheat_sheet.md",
        contains=("typed policy DSL", "aspf_opportunity_rules.yaml"),
    ),
    StatementPattern(
        path="docs/aspf_execution_fibration.md",
        contains=("Policy DSL ownership", "Lattice algebra ownership"),
    ),
    StatementPattern(
        path="docs/policy_dsl_migration_notes.md",
        contains=("boundary adapters", "removal_condition"),
    ),
)


def _sorted[T](values: Iterable[T], *, key=None) -> list[T]:
    return ordered_or_sorted(list(values), source="scripts.policy.build_projection_spec_history", key=key)


class GitInterface:
    def log(self, *, paths: tuple[str, ...]) -> tuple[CommitRecord, ...]:
        raise NotImplementedError

    def status(self) -> tuple[WorkspaceChange, ...]:
        raise NotImplementedError

    def commit_exists(self, sha: str) -> bool:
        raise NotImplementedError

    def tracked(self, path: str) -> bool:
        raise NotImplementedError


class ShellGit(GitInterface):
    def __init__(self, *, repo_root: Path) -> None:
        self._repo_root = repo_root

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self._repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

    def log(self, *, paths: tuple[str, ...]) -> tuple[CommitRecord, ...]:
        if not paths:
            return ()
        command = [
            "git",
            "log",
            "--date=short",
            "--pretty=format:%ad%x09%h%x09%s",
            "--",
            *paths,
        ]
        try:
            completed = self._run(command)
        except subprocess.CalledProcessError:
            return ()
        records: list[CommitRecord] = []
        seen: set[str] = set()
        for line in completed.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            date, sha, subject = parts
            if sha in seen:
                continue
            seen.add(sha)
            records.append(
                CommitRecord(
                    sha=sha.strip(),
                    date=date.strip(),
                    subject=subject.strip(),
                )
            )
        return tuple(
            _sorted(
                records,
                key=lambda item: (item.date, item.sha, item.subject),
            )
        )

    def status(self) -> tuple[WorkspaceChange, ...]:
        try:
            completed = self._run(["git", "status", "--porcelain"])
        except subprocess.CalledProcessError:
            return ()
        changes: list[WorkspaceChange] = []
        for line in completed.stdout.splitlines():
            if not line:
                continue
            status_code = line[:2]
            raw_path = line[3:].strip()
            path = raw_path.split(" -> ")[-1]
            changes.append(
                WorkspaceChange(
                    status_code=status_code,
                    path=path,
                )
            )
        return tuple(
            _sorted(
                changes,
                key=lambda item: (item.path, item.status_code),
            )
        )

    def commit_exists(self, sha: str) -> bool:
        try:
            _ = self._run(["git", "cat-file", "-e", f"{sha}^{{commit}}"])
            return True
        except subprocess.CalledProcessError:
            return False

    def tracked(self, path: str) -> bool:
        try:
            _ = self._run(["git", "ls-files", "--error-unmatch", "--", path])
            return True
        except subprocess.CalledProcessError:
            return False


def _path_matches_surface(*, path: str, surface: str) -> bool:
    if path == surface:
        return True
    return path.startswith(f"{surface.rstrip('/')}/")


def _top_hotspots(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    files = inventory.get("files")
    if not isinstance(files, list):
        return []
    hotspots: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        term_hits = item.get("term_hit_count")
        category = item.get("category")
        area_tags = item.get("area_tags")
        if not isinstance(path, str):
            continue
        if not isinstance(term_hits, int):
            continue
        if not isinstance(category, str):
            continue
        normalized_tags = (
            [str(tag) for tag in area_tags]
            if isinstance(area_tags, list)
            else []
        )
        hotspots.append(
            {
                "path": path,
                "term_hit_count": term_hits,
                "category": category,
                "area_tags": normalized_tags,
            }
        )
    ordered = _sorted(
        hotspots,
        key=lambda item: (
            -int(item["term_hit_count"]),
            str(item["path"]),
        ),
    )
    return ordered[:_MAX_TOP_HOTSPOTS]


def _inventory_summary(inventory: dict[str, Any]) -> dict[str, int]:
    summary = inventory.get("summary")
    if not isinstance(summary, dict):
        return {"file_count": 0, "hit_count": 0, "term_hit_count": 0}
    return {
        "file_count": int(summary.get("file_count", 0) or 0),
        "hit_count": int(summary.get("hit_count", 0) or 0),
        "term_hit_count": int(summary.get("term_hit_count", 0) or 0),
    }


def _extract_statements(*, repo_root: Path) -> list[dict[str, str]]:
    statements: list[dict[str, str]] = []
    for pattern in _STATEMENT_PATTERNS:
        path = repo_root / pattern.path
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        matched = 0
        for line_no, line in enumerate(lines, start=1):
            lowered = line.lower()
            if not any(term.lower() in lowered for term in pattern.contains):
                continue
            text = line.strip()
            if not text:
                continue
            statements.append(
                {
                    "source": f"{pattern.path}:{line_no}",
                    "statement": text,
                }
            )
            matched += 1
            if matched >= pattern.limit:
                break
    return statements


def _select_era_statements(
    *,
    intent_sources: tuple[str, ...],
    statements: list[dict[str, str]],
) -> list[dict[str, str]]:
    source_prefixes = tuple(f"{path}:" for path in intent_sources)
    selected = [
        item
        for item in statements
        if any(str(item["source"]).startswith(prefix) for prefix in source_prefixes)
    ]
    return _sorted(
        selected,
        key=lambda item: (str(item["source"]), str(item["statement"])),
    )


def _completion_focus(
    *,
    repo_root: Path,
    git: GitInterface,
) -> tuple[list[dict[str, str]], list[str]]:
    projection_rules_path = "docs/projection_fiber_rules.yaml"
    lattice_path = "src/gabion/analysis/aspf/aspf_lattice_algebra.py"
    policy_check_path = repo_root / "scripts/policy/policy_check.py"
    policy_check_source = (
        policy_check_path.read_text(encoding="utf-8")
        if policy_check_path.exists()
        else ""
    )
    lattice_source = (
        (repo_root / lattice_path).read_text(encoding="utf-8")
        if (repo_root / lattice_path).exists()
        else ""
    )
    substrate_init_path = repo_root / "src/gabion/tooling/policy_substrate/__init__.py"
    substrate_adapter_path = repo_root / "src/gabion/tooling/policy_substrate/dataflow_fibration.py"
    substrate_init_source = (
        substrate_init_path.read_text(encoding="utf-8")
        if substrate_init_path.exists()
        else ""
    )
    substrate_adapter_source = (
        substrate_adapter_path.read_text(encoding="utf-8")
        if substrate_adapter_path.exists()
        else ""
    )
    legacy_adapter_symbols = (
        "RecombinationFrontier",
        "compute_recombination_frontier",
        "empty_recombination_frontier",
    )
    adapter_free = (
        bool(substrate_init_source)
        and bool(substrate_adapter_source)
        and all(symbol not in substrate_init_source for symbol in legacy_adapter_symbols)
        and all(symbol not in substrate_adapter_source for symbol in legacy_adapter_symbols)
    )
    single_frontier_contract = (
        bool(lattice_source)
        and all(symbol not in lattice_source for symbol in legacy_adapter_symbols)
    )

    criteria = [
        {
            "criterion_id": "CF-01",
            "description": "ProjectionSpec inventory is present and populated",
            "status": "pass" if (repo_root / "artifacts/out/projection_spec_inventory.json").exists() else "fail",
            "evidence": "artifacts/out/projection_spec_inventory.json",
        },
        {
            "criterion_id": "CF-02",
            "description": "Projection-fiber DSL source is committed and registry-addressable",
            "status": "pass" if git.tracked(projection_rules_path) else "in_progress",
            "evidence": projection_rules_path,
        },
        {
            "criterion_id": "CF-03",
            "description": "Canonical lattice algebra module is committed",
            "status": "pass" if git.tracked(lattice_path) else "in_progress",
            "evidence": lattice_path,
        },
        {
            "criterion_id": "CF-04",
            "description": "Convergence gate is DSL/witness-only semantic evaluation",
            "status": (
                "pass"
                if (
                    "iter_semantic_lattice_convergence" in policy_check_source
                    and "materialize_semantic_lattice_convergence" in policy_check_source
                    and "collect_lattice_convergence_probe" not in policy_check_source
                )
                else "fail"
            ),
            "evidence": "scripts/policy/policy_check.py",
        },
        {
            "criterion_id": "CF-05",
            "description": "Policy DSL migration notes still constrain temporary boundary adapters",
            "status": "pass" if git.tracked("docs/policy_dsl_migration_notes.md") else "in_progress",
            "evidence": "docs/policy_dsl_migration_notes.md",
        },
        {
            "criterion_id": "CF-06",
            "description": "Policy substrate adapter exports are canonical witness-only",
            "status": "pass" if adapter_free else "in_progress",
            "evidence": (
                "src/gabion/tooling/policy_substrate/__init__.py; "
                "src/gabion/tooling/policy_substrate/dataflow_fibration.py"
            ),
        },
        {
            "criterion_id": "CF-07",
            "description": "Canonical lattice algebra uses a single FrontierWitness contract",
            "status": "pass" if single_frontier_contract else "in_progress",
            "evidence": lattice_path,
        },
    ]
    sequence = [
        "Commit projection-fiber rule source and lattice algebra as canonical tracked surfaces.",
        "Cut convergence checks to evaluator decisions over semantic lattice witnesses only.",
        "Eliminate remaining transitional frontier compatibility branches in policy substrate adapters.",
        "Lock deterministic lazy-pull and cache-parity tests as hard convergence gates.",
        "Enforce adapter-free substrate exports via drift checks and completion criteria.",
        "Enforce single frontier contract in canonical lattice algebra with no recombination bridge symbols.",
    ]
    return criteria, sequence


def _semantic_lowering_focus() -> dict[str, Any]:
    with deadline_scope(Deadline.from_timeout_ms(1000)):
        with deadline_clock_scope(GasMeter(limit=1_000_000)):
            rows = [
                _semantic_lowering_row(
                    lower_projection_spec_to_semantic_plan(spec)
                )
                for spec in _sorted(
                    list(iter_registered_specs()),
                    key=lambda item: (str(item.domain), str(item.name)),
                )
            ]
    summary = {
        "registered_spec_count": len(rows),
        "semantic_promoted_count": sum(
            1 for row in rows if int(row["semantic_op_count"]) > 0
        ),
        "presentation_only_count": sum(
            1
            for row in rows
            if row["lowering_status"] == "presentation_only"
        ),
        "bridge_present_count": sum(
            1 for row in rows if int(row["bridge_op_count"]) > 0
        ),
    }
    return {
        "summary": summary,
        "rows": rows,
    }


def _semantic_lowering_row(
    lowering_plan: ProjectionSemanticLoweringPlan,
) -> dict[str, Any]:
    semantic_ops = tuple(
        _semantic_op_payload(item) for item in lowering_plan.semantic_ops
    )
    presentation_ops = tuple(
        _presentation_op_payload(item) for item in lowering_plan.presentation_ops
    )
    bridge_ops = tuple(
        _bridge_op_payload(item) for item in lowering_plan.bridge_ops
    )
    quotient_faces = tuple(
        _sorted(
            [
                str(item["quotient_face"])
                for item in semantic_ops
                if isinstance(item.get("quotient_face"), str)
                and str(item["quotient_face"]).strip()
            ]
        )
    )
    lowering_status = _lowering_status(
        semantic_ops=semantic_ops,
        presentation_ops=presentation_ops,
        bridge_ops=bridge_ops,
    )
    return {
        "spec_identity": lowering_plan.spec_identity,
        "spec_name": lowering_plan.spec_name,
        "domain": lowering_plan.domain,
        "lowering_status": lowering_status,
        "semantic_op_count": len(semantic_ops),
        "presentation_op_count": len(presentation_ops),
        "bridge_op_count": len(bridge_ops),
        "quotient_faces": list(quotient_faces),
        "semantic_ops": [item for item in semantic_ops],
        "presentation_ops": [item for item in presentation_ops],
        "bridge_ops": [item for item in bridge_ops],
    }


def _semantic_op_payload(op: SemanticProjectionOp) -> dict[str, Any]:
    return {
        "source_index": op.source_index,
        "source_op": op.source_op,
        "semantic_op": op.semantic_op.value,
        "quotient_face": str(op.params.get("quotient_face", "") or ""),
    }


def _presentation_op_payload(op: PresentationProjectionOp) -> dict[str, Any]:
    return {
        "source_index": op.source_index,
        "source_op": op.source_op,
    }


def _bridge_op_payload(op: BridgeProjectionOp) -> dict[str, Any]:
    return {
        "source_index": op.source_index,
        "source_op": op.source_op,
        "bridge_kind": op.bridge_kind.value,
    }


def _lowering_status(
    *,
    semantic_ops: tuple[dict[str, Any], ...],
    presentation_ops: tuple[dict[str, Any], ...],
    bridge_ops: tuple[dict[str, Any], ...],
) -> str:
    has_semantic = bool(semantic_ops)
    has_presentation = bool(presentation_ops)
    has_bridge = bool(bridge_ops)
    if has_semantic and has_presentation:
        return "mixed"
    if has_semantic:
        return "semantic_promoted"
    if has_bridge and has_presentation:
        return "presentation_plus_bridge"
    if has_bridge:
        return "bridge_only"
    return "presentation_only"


def build_history(
    *,
    repo_root: Path,
    inventory: dict[str, Any],
    git: GitInterface,
) -> dict[str, Any]:
    summary = _inventory_summary(inventory)
    statements = _extract_statements(repo_root=repo_root)
    workspace_changes = git.status()
    eras: list[dict[str, Any]] = []

    for era_spec in _ERA_SPECS:
        lookup_paths = tuple(
            _sorted(
                [
                    *era_spec.intent_sources,
                    *era_spec.implementation_surfaces,
                    *era_spec.validation_surfaces,
                ]
            )
        )
        commits = git.log(paths=lookup_paths)
        date_start = commits[0].date if commits else ""
        date_end = commits[-1].date if commits else ""
        commit_evidence = commits
        if len(commit_evidence) > _MAX_COMMIT_EVIDENCE:
            half = _MAX_COMMIT_EVIDENCE // 2
            commit_evidence = (*commit_evidence[:half], *commit_evidence[-half:])
        workspace_delta = [
            {
                "path": change.path,
                "status_code": change.status_code,
                "provisional": True,
            }
            for change in workspace_changes
            if any(
                _path_matches_surface(path=change.path, surface=surface)
                for surface in era_spec.implementation_surfaces
            )
        ]
        status: EraStatus = era_spec.base_status
        if status != "open" and workspace_delta:
            status = "in_progress"

        eras.append(
            {
                "era_id": era_spec.era_id,
                "title": era_spec.title,
                "date_start": date_start,
                "date_end": date_end,
                "intent_sources": list(era_spec.intent_sources),
                "plan_statements": _select_era_statements(
                    intent_sources=era_spec.intent_sources,
                    statements=statements,
                ),
                "implementation_surfaces": list(era_spec.implementation_surfaces),
                "evidence_commits": [
                    {
                        "sha": record.sha,
                        "date": record.date,
                        "subject": record.subject,
                    }
                    for record in commit_evidence
                ],
                "validation_surfaces": list(era_spec.validation_surfaces),
                "status": status,
                "workspace_delta": workspace_delta,
                "completion_gaps": list(era_spec.completion_gaps),
                "next_actions": list(era_spec.next_actions),
            }
        )

    completion_criteria, closure_sequence = _completion_focus(
        repo_root=repo_root,
        git=git,
    )
    semantic_lowering_focus = _semantic_lowering_focus()
    return {
        "format_version": 1,
        "source": {
            "inventory_path": "artifacts/out/projection_spec_inventory.json",
            "inventory_generated_at_utc": inventory.get("generated_at_utc", ""),
            "root": str(repo_root),
        },
        "summary": {
            "file_count": summary["file_count"],
            "hit_count": summary["hit_count"],
            "term_hit_count": summary["term_hit_count"],
            "era_count": len(eras),
        },
        "top_hotspots": _top_hotspots(inventory),
        "eras": eras,
        "completion_focus": {
            "criteria": completion_criteria,
            "closure_sequence": closure_sequence,
        },
        "semantic_lowering_focus": semantic_lowering_focus,
    }


def validate_history(
    *,
    history: dict[str, Any],
    git: GitInterface,
    inventory: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    summary = history.get("summary")
    if not isinstance(summary, dict):
        return ["history.summary must be an object"]
    inventory_summary = _inventory_summary(inventory)
    if int(summary.get("file_count", -1)) != inventory_summary["file_count"]:
        errors.append("history summary file_count does not match inventory summary")
    if int(summary.get("term_hit_count", -1)) != inventory_summary["term_hit_count"]:
        errors.append("history summary term_hit_count does not match inventory summary")

    top_hotspots = history.get("top_hotspots")
    if not isinstance(top_hotspots, list):
        errors.append("history.top_hotspots must be a list")
    else:
        expected = _top_hotspots(inventory)
        if top_hotspots != expected:
            errors.append("history top_hotspots ordering does not match inventory ranking")

    semantic_lowering_focus = history.get("semantic_lowering_focus")
    if not isinstance(semantic_lowering_focus, dict):
        errors.append("history.semantic_lowering_focus must be an object")
    else:
        lowering_summary = semantic_lowering_focus.get("summary")
        if not isinstance(lowering_summary, dict):
            errors.append("history.semantic_lowering_focus.summary must be an object")
        lowering_rows = semantic_lowering_focus.get("rows")
        if not isinstance(lowering_rows, list):
            errors.append("history.semantic_lowering_focus.rows must be a list")
        else:
            required_lowering_fields = {
                "spec_identity",
                "spec_name",
                "domain",
                "lowering_status",
                "semantic_op_count",
                "presentation_op_count",
                "bridge_op_count",
                "quotient_faces",
                "semantic_ops",
                "presentation_ops",
                "bridge_ops",
            }
            for row in lowering_rows:
                if not isinstance(row, dict):
                    errors.append("history.semantic_lowering_focus.rows entries must be objects")
                    continue
                missing = required_lowering_fields - set(row.keys())
                if missing:
                    errors.append(
                        "semantic lowering row missing required fields: "
                        f"{sorted(missing)}"
                    )

    eras = history.get("eras")
    if not isinstance(eras, list):
        errors.append("history.eras must be a list")
        return errors

    required_fields = {
        "era_id",
        "title",
        "date_start",
        "date_end",
        "intent_sources",
        "plan_statements",
        "implementation_surfaces",
        "evidence_commits",
        "validation_surfaces",
        "status",
        "workspace_delta",
        "completion_gaps",
        "next_actions",
    }
    for era in eras:
        if not isinstance(era, dict):
            errors.append("history.eras entries must be objects")
            continue
        missing = required_fields - set(era.keys())
        if missing:
            errors.append(
                f"era {era.get('era_id', '<unknown>')} missing required fields: {sorted(missing)}"
            )
        commits = era.get("evidence_commits")
        if not isinstance(commits, list):
            errors.append(f"era {era.get('era_id', '<unknown>')} evidence_commits must be a list")
            continue
        for record in commits:
            if not isinstance(record, dict):
                errors.append(
                    f"era {era.get('era_id', '<unknown>')} contains invalid commit record"
                )
                continue
            sha = str(record.get("sha", "")).strip()
            if not sha:
                errors.append(
                    f"era {era.get('era_id', '<unknown>')} contains commit record with empty sha"
                )
                continue
            if not git.commit_exists(sha):
                errors.append(
                    f"era {era.get('era_id', '<unknown>')} references missing commit {sha}"
                )
        workspace_delta = era.get("workspace_delta")
        if isinstance(workspace_delta, list):
            for change in workspace_delta:
                if not isinstance(change, dict):
                    errors.append(
                        f"era {era.get('era_id', '<unknown>')} has invalid workspace_delta entry"
                    )
                    continue
                if change.get("provisional") is not True:
                    errors.append(
                        f"era {era.get('era_id', '<unknown>')} workspace_delta entry must be provisional=true"
                    )
    return errors


def render_markdown(history: dict[str, Any]) -> str:
    summary = history.get("summary", {})
    lines = [
        "---",
        "doc_revision: 1",
        "doc_id: projection_spec_history_ledger",
        "doc_role: audit",
        "---",
        "",
        "# ProjectionSpec Chronology Ledger",
        "",
        "## Summary",
        f"- file_count: {int(summary.get('file_count', 0) or 0)}",
        f"- hit_count: {int(summary.get('hit_count', 0) or 0)}",
        f"- term_hit_count: {int(summary.get('term_hit_count', 0) or 0)}",
        f"- era_count: {int(summary.get('era_count', 0) or 0)}",
        "",
        "## Top Hotspots",
        "",
        "| rank | path | term_hit_count | category | area_tags |",
        "| ---: | --- | ---: | --- | --- |",
    ]

    hotspots = history.get("top_hotspots", [])
    if isinstance(hotspots, list):
        for index, item in enumerate(hotspots, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {rank} | {path} | {hits} | {category} | {tags} |".format(
                    rank=index,
                    path=str(item.get("path", "")),
                    hits=int(item.get("term_hit_count", 0) or 0),
                    category=str(item.get("category", "")),
                    tags=", ".join(str(tag) for tag in item.get("area_tags", [])),
                )
            )

    eras = history.get("eras", [])
    if isinstance(eras, list):
        for era in eras:
            if not isinstance(era, dict):
                continue
            lines.extend(
                [
                    "",
                    f"## {era.get('era_id', '')}: {era.get('title', '')}",
                    f"- status: `{era.get('status', '')}`",
                    f"- date_window: `{era.get('date_start', '')}` -> `{era.get('date_end', '')}`",
                    "",
                    "### Intent At The Time",
                ]
            )
            statements = era.get("plan_statements", [])
            if isinstance(statements, list) and statements:
                for statement in statements:
                    if not isinstance(statement, dict):
                        continue
                    lines.append(
                        f"- {statement.get('statement', '')} ({statement.get('source', '')})"
                    )
            else:
                lines.append("- No explicit statement excerpt found in configured intent sources.")

            lines.append("")
            lines.append("### What Shipped")
            implementation = era.get("implementation_surfaces", [])
            if isinstance(implementation, list) and implementation:
                for path in implementation:
                    lines.append(f"- `{path}`")
            else:
                lines.append("- No committed implementation surface for this era yet.")

            commits = era.get("evidence_commits", [])
            lines.append("")
            lines.append("### Evidence Commits")
            if isinstance(commits, list) and commits:
                for record in commits:
                    if not isinstance(record, dict):
                        continue
                    lines.append(
                        f"- `{record.get('sha', '')}` `{record.get('date', '')}` {record.get('subject', '')}"
                    )
            else:
                lines.append("- No commit evidence captured for configured surfaces.")

            lines.append("")
            lines.append("### What Drifted")
            workspace_delta = era.get("workspace_delta", [])
            if isinstance(workspace_delta, list) and workspace_delta:
                for change in workspace_delta:
                    if not isinstance(change, dict):
                        continue
                    lines.append(
                        "- `[PROVISIONAL]` `{path}` ({status})".format(
                            path=change.get("path", ""),
                            status=change.get("status_code", ""),
                        )
                    )
            else:
                lines.append("- No provisional workspace-only delta for this era.")

            lines.append("")
            lines.append("### What Remains")
            gaps = era.get("completion_gaps", [])
            actions = era.get("next_actions", [])
            if isinstance(gaps, list) and gaps:
                for gap in gaps:
                    lines.append(f"- {gap}")
            else:
                lines.append("- No explicit completion gap recorded for this era.")
            if isinstance(actions, list) and actions:
                lines.append("")
                lines.append("### Next Actions")
                for action in actions:
                    lines.append(f"- {action}")

    focus = history.get("completion_focus", {})
    semantic_lowering_focus = history.get("semantic_lowering_focus", {})
    lines.extend(
        [
            "",
            "## Completion Focus Appendix",
            "",
            "### Substrate Convergence Criteria",
            "",
            "| criterion_id | description | status | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    criteria = focus.get("criteria")
    if isinstance(criteria, list):
        for item in criteria:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {cid} | {description} | {status} | {evidence} |".format(
                    cid=str(item.get("criterion_id", "")),
                    description=str(item.get("description", "")),
                    status=str(item.get("status", "")),
                    evidence=str(item.get("evidence", "")),
                )
            )

    lines.extend(["", "### Prioritized Closure Sequence"])
    sequence = focus.get("closure_sequence")
    if isinstance(sequence, list) and sequence:
        for index, step in enumerate(sequence, start=1):
            lines.append(f"{index}. {step}")
    else:
        lines.append("1. No closure sequence recorded.")

    lines.extend(
        [
            "",
            "## Semantic Lowering Appendix",
            "",
            "| spec_name | domain | status | semantic | presentation | bridge | quotient_faces |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    lowering_rows = semantic_lowering_focus.get("rows")
    if isinstance(lowering_rows, list):
        for row in lowering_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {spec_name} | {domain} | {status} | {semantic} | {presentation} | {bridge} | {faces} |".format(
                    spec_name=str(row.get("spec_name", "")),
                    domain=str(row.get("domain", "")),
                    status=str(row.get("lowering_status", "")),
                    semantic=int(row.get("semantic_op_count", 0) or 0),
                    presentation=int(row.get("presentation_op_count", 0) or 0),
                    bridge=int(row.get("bridge_op_count", 0) or 0),
                    faces=", ".join(str(item) for item in row.get("quotient_faces", [])),
                )
            )
    else:
        lines.append("| <none> |  |  | 0 | 0 | 0 |  |")

    lines.append("")
    return "\n".join(lines)


def run(
    *,
    repo_root: Path,
    inventory_path: Path,
    out_json_path: Path,
    out_markdown_path: Path,
) -> int:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    git = ShellGit(repo_root=repo_root)
    history = build_history(repo_root=repo_root, inventory=inventory, git=git)
    errors = validate_history(history=history, git=git, inventory=inventory)
    if errors:
        raise RuntimeError("; ".join(errors))

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    out_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    out_markdown_path.write_text(render_markdown(history), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build ProjectionSpec chronology ledger from inventory + repo evidence."
    )
    parser.add_argument(
        "--inventory",
        default="artifacts/out/projection_spec_inventory.json",
    )
    parser.add_argument(
        "--out-json",
        default="artifacts/out/projection_spec_history_ledger.json",
    )
    parser.add_argument(
        "--out-markdown",
        default="docs/audits/projection_spec_history_ledger.md",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    return run(
        repo_root=repo_root,
        inventory_path=(repo_root / args.inventory).resolve(),
        out_json_path=(repo_root / args.out_json).resolve(),
        out_markdown_path=(repo_root / args.out_markdown).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
