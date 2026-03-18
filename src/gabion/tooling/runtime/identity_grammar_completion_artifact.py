from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from gabion.order_contract import ordered_or_sorted

_ARTIFACT_KIND = "identity_grammar_completion"
_SCHEMA_VERSION = 1
_GENERATED_BY = (
    "gabion.tooling.runtime.identity_grammar_completion_artifact."
    "build_identity_grammar_completion_artifact_payload"
)

_HOTSPOT_QUEUE_PATH = "scripts/policy/hotspot_neighborhood_queue.py"
_PLANNING_CHART_IDENTITY_PATH = (
    "src/gabion/tooling/policy_substrate/planning_chart_identity.py"
)
_IDENTITY_GRAMMAR_PATH = (
    "src/gabion/tooling/policy_substrate/identity_zone/grammar.py"
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.runtime.identity_grammar_completion_artifact",
        key=key,
    )


class IdentityGrammarSurfacePayload(TypedDict):
    surface_id: str
    title: str
    status: str
    summary: str
    evidence_paths: list[str]
    residue_ids: list[str]


class IdentityGrammarResiduePayload(TypedDict):
    residue_id: str
    surface_id: str
    residue_kind: str
    severity: str
    score: int
    title: str
    message: str
    evidence_paths: list[str]


class IdentityGrammarCompletionArtifactPayload(TypedDict):
    artifact_kind: str
    schema_version: int
    generated_by: str
    summary: dict[str, object]
    surfaces: list[IdentityGrammarSurfacePayload]
    residues: list[IdentityGrammarResiduePayload]


@dataclass(frozen=True)
class _SurfaceSpec:
    surface_id: str
    title: str
    summary: str
    evidence_paths: tuple[str, ...]
    residue_kind: str
    severity: str
    score: int
    message: str


@dataclass(frozen=True)
class _EvaluatedSurface:
    spec: _SurfaceSpec
    status: str
    residue_id: str


_SURFACE_SPECS = (
    _SurfaceSpec(
        surface_id="identity_grammar.hotspot.raw_string_grouping",
        title="Hotspot queue still groups by raw path strings",
        summary=(
            "Core hotspot grouping should run on typed carriers or quotient-linked "
            "carriers, not raw rel_path strings."
        ),
        evidence_paths=(_HOTSPOT_QUEUE_PATH,),
        residue_kind="raw_string_grouping_in_core_queue_logic",
        severity="high",
        score=9,
        message=(
            "hotspot_neighborhood_queue._file_family_counts still groups violations "
            "through raw path-keyed dictionaries before constructing hotspot carriers."
        ),
    ),
    _SurfaceSpec(
        surface_id="identity_grammar.hotspot.file_quotient",
        title="Hotspot file quotient only reifies the chosen representative",
        summary=(
            "File-level quotient witnesses should cover the full upstream fiber, not "
            "only the chosen representative wire."
        ),
        evidence_paths=(_HOTSPOT_QUEUE_PATH,),
        residue_kind="partial_file_quotient_reification",
        severity="high",
        score=8,
        message=(
            "hotspot_neighborhood_queue._file_ref records the full fiber but binds "
            "file quotient and zone morphism edges only for the chosen representative wire."
        ),
    ),
    _SurfaceSpec(
        surface_id="identity_grammar.hotspot.scope_quotient",
        title="Hotspot scope quotient only reifies the chosen representative",
        summary=(
            "Scope-level quotient witnesses should cover the full upstream fiber, not "
            "only the chosen representative wire."
        ),
        evidence_paths=(_HOTSPOT_QUEUE_PATH,),
        residue_kind="partial_scope_quotient_reification",
        severity="high",
        score=8,
        message=(
            "hotspot_neighborhood_queue._scope_ref records the full fiber but binds "
            "scope quotient and zone morphism edges only for the chosen representative wire."
        ),
    ),
    _SurfaceSpec(
        surface_id="identity_grammar.planning_chart.integration",
        title="Planning-chart identity grammar is not wired into the live planner",
        summary=(
            "The planning chart zone should be constructed inside a production planning "
            "path, not only in tests."
        ),
        evidence_paths=(_PLANNING_CHART_IDENTITY_PATH,),
        residue_kind="planning_chart_identity_grammar_unintegrated",
        severity="medium",
        score=7,
        message=(
            "build_planning_chart_identity_grammar currently has no production callers "
            "outside tests and re-export surfaces."
        ),
    ),
    _SurfaceSpec(
        surface_id="identity_grammar.coherence.two_cell",
        title="Coherence witness carrier exists but is not emitted",
        summary=(
            "The identity grammar should emit live 2-cell/coherence witnesses, not just "
            "define the carrier type."
        ),
        evidence_paths=(_IDENTITY_GRAMMAR_PATH,),
        residue_kind="coherence_witness_emission_missing",
        severity="medium",
        score=6,
        message=(
            "HierarchicalIdentityGrammar.add_two_cell exists, but there are no production "
            "emitters for coherence witnesses."
        ),
    ),
)


def _read_source(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def _parse_module(root: Path, rel_path: str) -> tuple[ast.AST, list[str]] | None:
    path = root / rel_path
    try:
        source = _read_source(root, rel_path)
    except OSError:
        return None
    try:
        return (ast.parse(source, filename=str(path)), source.splitlines())
    except SyntaxError:
        return None


def _qualname_segment(root: Path, rel_path: str, qualname: str) -> str:
    parsed = _parse_module(root, rel_path)
    if parsed is None:
        return ""
    module, lines = parsed
    parts = qualname.split(".")
    nodes: list[ast.stmt] = list(getattr(module, "body", ()))
    current: ast.AST | None = None
    for part in parts:
        match = None
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == part:
                    match = node
                    break
        if match is None:
            return ""
        current = match
        nodes = list(getattr(match, "body", ()))
    if current is None:
        return ""
    start = int(getattr(current, "lineno", 0) or 0)
    end = int(getattr(current, "end_lineno", start) or start)
    if start <= 0 or end < start:
        return ""
    return "\n".join(lines[start - 1 : end])


def _count_symbol_calls(
    root: Path,
    *,
    symbol_name: str,
    exclude_rel_paths: tuple[str, ...] = (),
) -> int:
    src_root = root / "src" / "gabion"
    if not src_root.exists():
        return 0
    count = 0
    for path in src_root.rglob("*.py"):
        rel_path = path.relative_to(root).as_posix()
        if rel_path in exclude_rel_paths:
            continue
        parsed = _parse_module(root, rel_path)
        if parsed is None:
            continue
        module, _ = parsed
        for node in filter(_is_call_node, ast.walk(module)):
            func = node.func
            if isinstance(func, ast.Name) and func.id == symbol_name:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr == symbol_name:
                count += 1
    return count


def _is_call_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Call)


def _has_raw_string_grouping(root: Path) -> bool:
    segment = _qualname_segment(root, _HOTSPOT_QUEUE_PATH, "_file_family_counts")
    return all(
        token in segment
        for token in (
            "counts_by_path",
            "source_carriers_by_path",
            "counts_by_path[path][family]",
            "for path, counts in counts_by_path.items()",
        )
    )


def _has_representative_only_quotient(root: Path, *, qualname: str) -> bool:
    segment = _qualname_segment(root, _HOTSPOT_QUEUE_PATH, qualname)
    return all(
        token in segment
        for token in (
            "member_source_wires",
            "chosen_wire",
            "source_carrier_wire=chosen_wire or identity.wire()",
            "add_quotient_projection",
            "add_zone_morphism",
        )
    )


def _planning_chart_grammar_is_unintegrated(root: Path) -> bool:
    return (
        _count_symbol_calls(
            root,
            symbol_name="build_planning_chart_identity_grammar",
            exclude_rel_paths=(_PLANNING_CHART_IDENTITY_PATH,),
        )
        == 0
    )


def _coherence_witness_emission_is_missing(root: Path) -> bool:
    return (
        _count_symbol_calls(
            root,
            symbol_name="add_two_cell",
            exclude_rel_paths=(_IDENTITY_GRAMMAR_PATH,),
        )
        == 0
    )


def _evaluate_surface(root: Path, spec: _SurfaceSpec) -> _EvaluatedSurface:
    match spec.surface_id:
        case "identity_grammar.hotspot.raw_string_grouping":
            failing = _has_raw_string_grouping(root)
        case "identity_grammar.hotspot.file_quotient":
            failing = _has_representative_only_quotient(root, qualname="_file_ref")
        case "identity_grammar.hotspot.scope_quotient":
            failing = _has_representative_only_quotient(root, qualname="_scope_ref")
        case "identity_grammar.planning_chart.integration":
            failing = _planning_chart_grammar_is_unintegrated(root)
        case "identity_grammar.coherence.two_cell":
            failing = _coherence_witness_emission_is_missing(root)
        case _:
            failing = False
    return _EvaluatedSurface(
        spec=spec,
        status="fail" if failing else "pass",
        residue_id=f"{spec.surface_id}:{spec.residue_kind}",
    )


def _highest_severity(residues: list[IdentityGrammarResiduePayload]) -> str:
    if not residues:
        return "pass"
    priority = {"critical": 4, "high": 3, "medium": 2, "warning": 1, "low": 0}
    return max(residues, key=lambda item: priority.get(item["severity"], -1))[
        "severity"
    ]


def build_identity_grammar_completion_artifact_payload(
    *,
    root: Path,
) -> IdentityGrammarCompletionArtifactPayload:
    evaluated = [_evaluate_surface(root, spec) for spec in _SURFACE_SPECS]
    residues: list[IdentityGrammarResiduePayload] = []
    surfaces: list[IdentityGrammarSurfacePayload] = []
    for item in evaluated:
        if item.status != "pass":
            residues.append(
                {
                    "residue_id": item.residue_id,
                    "surface_id": item.spec.surface_id,
                    "residue_kind": item.spec.residue_kind,
                    "severity": item.spec.severity,
                    "score": item.spec.score,
                    "title": item.spec.title,
                    "message": item.spec.message,
                    "evidence_paths": list(item.spec.evidence_paths),
                }
            )
        surfaces.append(
            {
                "surface_id": item.spec.surface_id,
                "title": item.spec.title,
                "status": item.status,
                "summary": item.spec.summary,
                "evidence_paths": list(item.spec.evidence_paths),
                "residue_ids": [] if item.status == "pass" else [item.residue_id],
            }
        )
    ordered_residues = _sorted(residues, key=lambda item: (-int(item["score"]), item["residue_id"]))
    ordered_surfaces = _sorted(surfaces, key=lambda item: item["surface_id"])
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "summary": {
            "surface_count": len(ordered_surfaces),
            "pass_count": sum(1 for item in ordered_surfaces if item["status"] == "pass"),
            "fail_count": sum(1 for item in ordered_surfaces if item["status"] != "pass"),
            "residue_count": len(ordered_residues),
            "highest_severity": _highest_severity(ordered_residues),
        },
        "surfaces": ordered_surfaces,
        "residues": ordered_residues,
    }


def write_identity_grammar_completion_artifact(*, path: Path, root: Path) -> Path:
    payload = build_identity_grammar_completion_artifact_payload(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "build_identity_grammar_completion_artifact_payload",
    "write_identity_grammar_completion_artifact",
]
