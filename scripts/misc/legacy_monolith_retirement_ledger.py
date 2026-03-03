#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    domain_id: str
    legacy_surface: str
    replacement_surface: str
    replacement_module: str
    proof_kind: str
    capability: str
    substitute: str
    required_surfaces: tuple[str, ...]
    intentional_drift: bool = False


_CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        capability_id="d1_path_resolution",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith.resolve_analysis_paths",
        replacement_surface="dataflow_ingest_helpers.resolve_analysis_paths",
        replacement_module="src/gabion/analysis/dataflow_ingest_helpers.py",
        proof_kind="surface_parity",
        capability="Path resolution",
        substitute="gabion check raw path normalization",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d1_function_collection",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith._collect_functions",
        replacement_surface="dataflow_ingest_helpers._collect_functions",
        replacement_module="src/gabion/analysis/dataflow_ingest_helpers.py",
        proof_kind="surface_parity",
        capability="Function collection",
        substitute="function-index projections",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d1_symbol_table",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith._build_symbol_table",
        replacement_surface="dataflow_evidence_helpers._build_symbol_table",
        replacement_module="src/gabion/analysis/dataflow_evidence_helpers.py",
        proof_kind="surface_parity",
        capability="Symbol table indexing",
        substitute="call-resolution evidence projection",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d1_class_index",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith._collect_class_index",
        replacement_surface="dataflow_evidence_helpers._collect_class_index",
        replacement_module="src/gabion/analysis/dataflow_evidence_helpers.py",
        proof_kind="surface_parity",
        capability="Class indexing",
        substitute="call-resolution evidence projection",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d1_function_index",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith._build_function_index",
        replacement_surface="dataflow_function_index_helpers._build_function_index",
        replacement_module="src/gabion/analysis/dataflow_function_index_helpers.py",
        proof_kind="surface_parity",
        capability="Function index + lambda indexing",
        substitute="call-resolution evidence projection",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d1_indexed_file_scan_cluster",
        domain_id="D1",
        legacy_surface="legacy_dataflow_monolith indexed scan helpers",
        replacement_surface="dataflow_indexed_file_scan helper cluster",
        replacement_module="src/gabion/analysis/dataflow_indexed_file_scan.py",
        proof_kind="surface_parity",
        capability="Indexed file scan helper cluster ownership",
        substitute="indexed pass projections used by pipeline domains",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d2_bundle_inference",
        domain_id="D2",
        legacy_surface="legacy_dataflow_monolith.analyze_paths bundle propagation",
        replacement_surface="gabion check raw groups_by_path projection",
        replacement_module="src/gabion/analysis/dataflow_pipeline.py",
        proof_kind="surface_parity",
        capability="Bundle inference and propagation",
        substitute="mise exec -- python -m gabion check raw -- <paths>",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d2_bundle_forest",
        domain_id="D2",
        legacy_surface="legacy_dataflow_monolith bundle forest population",
        replacement_surface="forest-backed bundle projection",
        replacement_module="src/gabion/analysis/dataflow_bundle_iteration.py",
        proof_kind="surface_parity",
        capability="Bundle forest population",
        substitute="ASPF forest + groups_by_path parity",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d2_resume_payload",
        domain_id="D2",
        legacy_surface="legacy_dataflow_monolith resume payload/readback",
        replacement_surface="analysis resume payload helpers",
        replacement_module="src/gabion/analysis/dataflow_analysis_index.py",
        proof_kind="metamorphic_commutation",
        capability="Resume payload/readback",
        substitute="aspf_state + delta ledger continuity",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d2_call_resolution_graph",
        domain_id="D2",
        legacy_surface="legacy_dataflow_monolith call graph/candidate materialization",
        replacement_surface="extracted obligations/callee graph surfaces",
        replacement_module="src/gabion/analysis/dataflow_obligations.py",
        proof_kind="surface_parity",
        capability="Call-resolution graph materialization",
        substitute="call ambiguity and obligation surfaces",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d3_decision_surfaces",
        domain_id="D3",
        legacy_surface="legacy_dataflow_monolith decision surfaces",
        replacement_surface="decision surface extraction + ASPF surface",
        replacement_module="src/gabion/analysis/dataflow_decision_surfaces.py",
        proof_kind="surface_parity",
        capability="Decision surfaces",
        substitute="ASPF decision_surfaces",
        required_surfaces=("decision_surfaces",),
    ),
    CapabilitySpec(
        capability_id="d3_value_decision_surfaces",
        domain_id="D3",
        legacy_surface="legacy_dataflow_monolith value decision surfaces",
        replacement_surface="decision surface extraction + ASPF value surface",
        replacement_module="src/gabion/analysis/dataflow_decision_surfaces.py",
        proof_kind="surface_parity",
        capability="Value decision surfaces",
        substitute="ASPF value_decision_surfaces",
        required_surfaces=("value_decision_surfaces",),
    ),
    CapabilitySpec(
        capability_id="d3_ambiguity_suite",
        domain_id="D3",
        legacy_surface="legacy_dataflow_monolith ambiguity suite rows",
        replacement_surface="ambiguity helper projection rows",
        replacement_module="src/gabion/analysis/dataflow_ambiguity_helpers.py",
        proof_kind="surface_parity",
        capability="Ambiguity suite and virtual set rows",
        substitute="ASPF decision/value surfaces + ambiguity projections",
        required_surfaces=("decision_surfaces", "value_decision_surfaces"),
    ),
    CapabilitySpec(
        capability_id="d3_pattern_instances",
        domain_id="D3",
        legacy_surface="legacy_dataflow_monolith pattern schema instances",
        replacement_surface="ASPF pattern_schema_instances",
        replacement_module="src/gabion/analysis/pattern_schema_projection.py",
        proof_kind="metamorphic_commutation",
        capability="PatternSchema instances",
        substitute=(
            "mise exec -- python -m gabion check raw -- "
            "--aspf-semantic-surface pattern_schema_instances"
        ),
        required_surfaces=("pattern_schema_instances",),
    ),
    CapabilitySpec(
        capability_id="d3_pattern_residue",
        domain_id="D3",
        legacy_surface="legacy_dataflow_monolith pattern schema residue",
        replacement_surface="ASPF pattern_schema_residue",
        replacement_module="src/gabion/analysis/pattern_schema_projection.py",
        proof_kind="metamorphic_commutation",
        capability="PatternSchema residue",
        substitute=(
            "mise exec -- python -m gabion check raw -- "
            "--aspf-semantic-surface pattern_schema_residue"
        ),
        required_surfaces=("pattern_schema_residue",),
    ),
    CapabilitySpec(
        capability_id="d4_fingerprint_matches",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith fingerprint matches",
        replacement_surface="fingerprint helper matches surface",
        replacement_module="src/gabion/analysis/dataflow_fingerprint_helpers.py",
        proof_kind="surface_parity",
        capability="Fingerprint matches",
        substitute="ASPF rewrite_plans + violation summary",
        required_surfaces=("rewrite_plans",),
    ),
    CapabilitySpec(
        capability_id="d4_fingerprint_provenance",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith fingerprint provenance",
        replacement_surface="fingerprint helper provenance surface",
        replacement_module="src/gabion/analysis/dataflow_fingerprint_helpers.py",
        proof_kind="surface_parity",
        capability="Fingerprint provenance",
        substitute="ASPF rewrite_plans provenance",
        required_surfaces=("rewrite_plans",),
    ),
    CapabilitySpec(
        capability_id="d4_fingerprint_synth",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith fingerprint synth",
        replacement_surface="fingerprint helper synth surface",
        replacement_module="src/gabion/analysis/dataflow_fingerprint_helpers.py",
        proof_kind="surface_parity",
        capability="Fingerprint synth",
        substitute="ASPF rewrite_plans synthesis projection",
        required_surfaces=("rewrite_plans",),
    ),
    CapabilitySpec(
        capability_id="d4_coherence",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith coherence witnesses",
        replacement_surface="fingerprint helper coherence witness surface",
        replacement_module="src/gabion/analysis/dataflow_fingerprint_helpers.py",
        proof_kind="surface_parity",
        capability="Coherence witnesses",
        substitute="ASPF rewrite_plans/coherence surface",
        required_surfaces=("rewrite_plans",),
    ),
    CapabilitySpec(
        capability_id="d4_rewrite_plans",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith rewrite plan emission",
        replacement_surface="fingerprint helper rewrite plans",
        replacement_module="src/gabion/analysis/dataflow_fingerprint_helpers.py",
        proof_kind="surface_parity",
        capability="Rewrite plans",
        substitute="ASPF rewrite_plans",
        required_surfaces=("rewrite_plans",),
    ),
    CapabilitySpec(
        capability_id="d4_structure_reuse",
        domain_id="D4",
        legacy_surface="legacy_dataflow_monolith structure reuse + lemma stubs",
        replacement_surface="structure reuse extracted surfaces",
        replacement_module="src/gabion/analysis/dataflow_structure_reuse.py",
        proof_kind="surface_parity",
        capability="Structure reuse + lemma stubs",
        substitute="structure snapshot + reuse projection",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d5_deadline_obligations",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith deadline obligations",
        replacement_surface="dataflow_obligations.collect_deadline_obligations",
        replacement_module="src/gabion/analysis/dataflow_obligations.py",
        proof_kind="surface_parity",
        capability="Deadline obligations",
        substitute="ASPF violation_summary + deadline projection",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_deadline_helper_ownership",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith deadline helper cluster",
        replacement_surface="dataflow_deadline_helpers helper surfaces",
        replacement_module="src/gabion/analysis/dataflow_deadline_helpers.py",
        proof_kind="surface_parity",
        capability="Deadline helper cluster ownership",
        substitute="obligation helper parity over extracted deadline helpers",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_exception_obligations",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith exception obligations",
        replacement_surface="extracted exception obligation surfaces",
        replacement_module="src/gabion/analysis/dataflow_exception_obligations.py",
        proof_kind="surface_parity",
        capability="Exception obligations",
        substitute="ASPF violation_summary",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_handledness_witnesses",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith handledness witnesses",
        replacement_surface="extracted handledness witness surfaces",
        replacement_module="src/gabion/analysis/dataflow_exception_obligations.py",
        proof_kind="surface_parity",
        capability="Handledness witnesses",
        substitute="ASPF violation_summary",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_never_invariants",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith never invariants",
        replacement_surface="never invariant extracted surfaces",
        replacement_module="src/gabion/analysis/dataflow_obligations.py",
        proof_kind="surface_parity",
        capability="Never invariants",
        substitute="ASPF violation_summary",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_lint_lines",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith lint-line materialization",
        replacement_surface="lint helper extraction",
        replacement_module="src/gabion/analysis/dataflow_decision_surfaces.py",
        proof_kind="surface_parity",
        capability="Lint lines",
        substitute="ASPF violation_summary",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d5_lint_helper_ownership",
        domain_id="D5",
        legacy_surface="legacy_dataflow_monolith lint helper cluster",
        replacement_surface="dataflow_lint_helpers helper surfaces",
        replacement_module="src/gabion/analysis/dataflow_lint_helpers.py",
        proof_kind="surface_parity",
        capability="Lint helper cluster ownership",
        substitute="lint witness/smell projection parity",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d6_report_render",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith render_report",
        replacement_surface="dataflow_reporting.render_report",
        replacement_module="src/gabion/analysis/dataflow_reporting.py",
        proof_kind="surface_parity",
        capability="Markdown report rendering",
        substitute="mise exec -- python -m gabion check raw -- --report <path>",
        required_surfaces=("violation_summary", "rewrite_plans"),
    ),
    CapabilitySpec(
        capability_id="d6_dot_render",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith render_dot",
        replacement_surface="dataflow_graph_rendering.render_dot",
        replacement_module="src/gabion/analysis/dataflow_graph_rendering.py",
        proof_kind="surface_parity",
        capability="DOT rendering",
        substitute="mise exec -- python -m gabion check raw -- --dot <path>",
        required_surfaces=("groups_by_path", "decision_surfaces"),
    ),
    CapabilitySpec(
        capability_id="d6_structure_snapshot",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith structure snapshot/load/diff",
        replacement_surface="dataflow_snapshot_io structure snapshot surfaces",
        replacement_module="src/gabion/analysis/dataflow_snapshot_io.py",
        proof_kind="surface_parity",
        capability="Structure snapshot/load/diff",
        substitute="emit/load/diff structure snapshot",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d6_decision_snapshot",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith decision snapshot/load/diff",
        replacement_surface="dataflow_snapshot_io decision snapshot surfaces",
        replacement_module="src/gabion/analysis/dataflow_snapshot_io.py",
        proof_kind="surface_parity",
        capability="Decision snapshot/load/diff",
        substitute="emit/load/diff decision snapshot",
        required_surfaces=("decision_surfaces", "pattern_schema_instances", "pattern_schema_residue"),
    ),
    CapabilitySpec(
        capability_id="d6_baseline_gate",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith baseline apply/load/write",
        replacement_surface="dataflow_baseline_gates baseline surfaces",
        replacement_module="src/gabion/analysis/dataflow_baseline_gates.py",
        proof_kind="gate_parity",
        capability="Baseline ratchet gate",
        substitute="baseline-mode write/enforce parity",
        required_surfaces=("violation_summary",),
    ),
    CapabilitySpec(
        capability_id="d6_run_outputs",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith sidecar/projection/run-output gating",
        replacement_surface="dataflow_run_outputs finalize_run_outputs",
        replacement_module="src/gabion/analysis/dataflow_run_outputs.py",
        proof_kind="surface_parity",
        capability="Run output gating and sidecar emission",
        substitute="check raw output sequence parity",
        required_surfaces=("violation_summary", "rewrite_plans"),
    ),
    CapabilitySpec(
        capability_id="d6_synthesis_refactor",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith synthesis/refactor plan+render",
        replacement_surface="dataflow_synthesis + dataflow_refactor_planning",
        replacement_module="src/gabion/analysis/dataflow_synthesis.py",
        proof_kind="surface_parity",
        capability="Synthesis/refactor planning and rendering",
        substitute="--synthesis* and --refactor-plan* parity",
        required_surfaces=("groups_by_path",),
    ),
    CapabilitySpec(
        capability_id="d6_raw_runtime_entry",
        domain_id="D6",
        legacy_surface="legacy_dataflow_monolith raw runtime entrypoints",
        replacement_surface="dataflow_raw_runtime entrypoints",
        replacement_module="src/gabion/analysis/dataflow_raw_runtime.py",
        proof_kind="gate_parity",
        capability="Raw runtime parser/run/main",
        substitute="gabion check raw parity",
        required_surfaces=("groups_by_path",),
    ),
)

_RUNTIME_IMPORT_RE = re.compile(
    r"^\s*(from gabion\.analysis\.legacy_dataflow_monolith import|"
    r"import gabion\.analysis\.legacy_dataflow_monolith|"
    r"from gabion\.analysis import legacy_dataflow_monolith|"
    r"from \.legacy_dataflow_monolith import|"
    r"import \.legacy_dataflow_monolith|"
    r"from \. import legacy_dataflow_monolith)\b",
    re.MULTILINE,
)

_PROBE_ARTIFACT_ROOT = "artifacts/out/runtime_retirement_probe"
_PROBE_STATE_SNAPSHOT = f"{_PROBE_ARTIFACT_ROOT}/aspf_state.snapshot.json"
_PROBE_DELTA_JSONL = f"{_PROBE_ARTIFACT_ROOT}/aspf_state.delta.jsonl"
_PROBE_REQUIRED_SURFACES: tuple[str, ...] = (
    "groups_by_path",
    "decision_surfaces",
    "value_decision_surfaces",
    "rewrite_plans",
    "violation_summary",
    "pattern_schema_instances",
    "pattern_schema_residue",
)


@dataclass(frozen=True)
class ShimDebtRow:
    surface: str
    owner: str
    rationale: str
    removal_correction_unit: str
    expiry: str
    exit_criteria: str
    status: str


def _runtime_import_count(*, root: Path, module_path: str) -> int:
    candidate = (root / module_path).resolve()
    if not candidate.exists():
        return 0
    text = candidate.read_text(encoding="utf-8")
    return len(_RUNTIME_IMPORT_RE.findall(text))


def _runtime_import_count_for_file(path: Path) -> int:
    if not path.exists():
        return 0
    return len(_RUNTIME_IMPORT_RE.findall(path.read_text(encoding="utf-8")))


def _resolve_analysis_module_path(*, root: Path, module_parts: tuple[str, ...]) -> Path | None:
    if len(module_parts) < 2:
        return None
    if module_parts[:2] != ("gabion", "analysis"):
        return None
    src_root = (root / "src").resolve()
    module_path = src_root.joinpath(*module_parts).with_suffix(".py")
    if module_path.exists():
        return module_path
    package_init = src_root.joinpath(*module_parts, "__init__.py")
    if package_init.exists():
        return package_init
    return None


def _analysis_module_dependencies(*, root: Path, module_file: Path) -> set[Path]:
    if not module_file.exists():
        return set()
    text = module_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    src_root = (root / "src").resolve()
    try:
        relative_from_src = module_file.resolve().relative_to(src_root)
    except ValueError:
        return set()
    current_module_parts = relative_from_src.with_suffix("").parts
    deps: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                resolved = _resolve_analysis_module_path(root=root, module_parts=parts)
                if resolved is not None:
                    deps.add(resolved)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level > 0:
            base = list(current_module_parts)
            for _ in range(node.level):
                if base:
                    base.pop()
            if node.module:
                base.extend(node.module.split("."))
            base_parts = tuple(base)
        else:
            base_parts = tuple(node.module.split(".")) if node.module else ()

        if base_parts:
            resolved_base = _resolve_analysis_module_path(root=root, module_parts=base_parts)
            if resolved_base is not None:
                deps.add(resolved_base)

        if base_parts == ("gabion", "analysis") or (
            node.level > 0 and node.module is None and base_parts[:2] == ("gabion", "analysis")
        ):
            for alias in node.names:
                target_parts = (*base_parts, alias.name)
                resolved_target = _resolve_analysis_module_path(
                    root=root,
                    module_parts=target_parts,
                )
                if resolved_target is not None:
                    deps.add(resolved_target)
    deps.discard(module_file.resolve())
    return deps


def _transitive_runtime_import_count(*, root: Path, module_path: str) -> int:
    start = (root / module_path).resolve()
    if not start.exists():
        return 0
    stack = [start]
    seen: set[Path] = set()
    total = 0
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _runtime_import_count_for_file(current)
        for dep in sorted(
            _analysis_module_dependencies(root=root, module_file=current),
            key=lambda path: str(path),
        ):
            if dep not in seen:
                stack.append(dep)
    return total


def _load_shim_lifecycle_rows_from_md(path: Path | None) -> list[ShimDebtRow]:
    if path is None or not path.exists():
        return []
    rows: list[ShimDebtRow] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("| ---"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 7:
            continue
        if columns[0] == "Surface":
            continue
        rows.append(
            ShimDebtRow(
                surface=columns[0],
                owner=columns[1],
                rationale=columns[2],
                removal_correction_unit=columns[3],
                expiry=columns[4],
                exit_criteria=columns[5],
                status=columns[6],
            )
        )
    return rows


def _shim_row_for_spec(
    *,
    spec: CapabilitySpec,
    shim_rows_by_capability: Mapping[str, dict[str, object]],
    shim_debt_rows: list[ShimDebtRow],
) -> dict[str, object] | None:
    if spec.capability_id in shim_rows_by_capability:
        return dict(shim_rows_by_capability[spec.capability_id])

    module_path = spec.replacement_module
    module_name = Path(module_path).name
    for row in shim_debt_rows:
        if row.status.lower() != "open":
            continue
        surface = row.surface
        if module_path in surface or module_name in surface:
            return {
                "actor": row.owner,
                "rationale": row.rationale,
                "scope": row.surface,
                "start": "",
                "expiry": row.expiry,
                "rollback_condition": row.exit_criteria,
                "evidence_links": ["docs/compatibility_layer_debt_register.md"],
                "removal_correction_unit": row.removal_correction_unit,
                "status": row.status,
            }
    return None


def _load_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        return {str(key): loaded[key] for key in loaded}
    raise ValueError(f"JSON payload at {path} must be an object")


def _load_jsonl(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        loaded = json.loads(line)
        if not isinstance(loaded, dict):
            raise ValueError(f"JSONL record in {path} must be an object")
        records.append({str(key): loaded[key] for key in loaded})
    return records


def _surface_classification_map(
    equivalence_payload: Mapping[str, object],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    raw_table = equivalence_payload.get("surface_table")
    if not isinstance(raw_table, list):
        return mapping
    for entry in raw_table:
        if not isinstance(entry, Mapping):
            continue
        surface = str(entry.get("surface", "") or "").strip()
        if not surface:
            continue
        classification = str(entry.get("classification", "") or "").strip()
        if classification:
            mapping[surface] = classification
    return mapping


def _validate_required_witness_surfaces(
    *,
    state_payload: Mapping[str, object],
    equivalence_payload: Mapping[str, object],
) -> None:
    state_surfaces = _state_semantic_surfaces(state_payload)
    equivalence_classes = _surface_classification_map(equivalence_payload)
    missing_state = [
        surface for surface in _PROBE_REQUIRED_SURFACES if surface not in state_surfaces
    ]
    missing_equivalence = [
        surface
        for surface in _PROBE_REQUIRED_SURFACES
        if surface not in equivalence_classes
    ]
    if missing_state or missing_equivalence:
        details = []
        if missing_state:
            details.append(
                "missing_state_surfaces=" + ",".join(sorted(missing_state))
            )
        if missing_equivalence:
            details.append(
                "missing_equivalence_surfaces="
                + ",".join(sorted(missing_equivalence))
            )
        raise ValueError(
            "retirement ledger witness validation failed: "
            + "; ".join(details)
            + ". Use the deterministic runtime-retirement probe with full surfaces."
        )


def _surface_delta_event_counts(
    *,
    ledger_records: list[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in ledger_records:
        target = str(record.get("mutation_target", "") or "").strip()
        if not target.startswith("semantic_surfaces."):
            continue
        surface = target.split(".", 1)[1]
        counts[surface] = counts.get(surface, 0) + 1
    return counts


def _state_semantic_surfaces(state_payload: Mapping[str, object]) -> dict[str, object]:
    raw = state_payload.get("semantic_surfaces")
    if isinstance(raw, Mapping):
        return {str(key): raw[key] for key in raw}
    return {}


def _digest_payload(*, payloads: Mapping[str, object]) -> str:
    serialized = json.dumps(payloads, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _status_for_capability(
    *,
    spec: CapabilitySpec,
    state_surfaces: Mapping[str, object],
    equivalence_classes: Mapping[str, str],
) -> tuple[str, str]:
    missing_state = [surface for surface in spec.required_surfaces if surface not in state_surfaces]
    drifted = [
        surface
        for surface in spec.required_surfaces
        if equivalence_classes.get(surface) not in {"", "non_drift"}
    ]
    if missing_state and not spec.intentional_drift:
        return "blocked", f"missing_state_surfaces={','.join(sorted(missing_state))}"
    if drifted and not spec.intentional_drift:
        return "blocked", f"equivalence_drift_surfaces={','.join(sorted(drifted))}"
    if missing_state or drifted:
        return "intentional_drift", (
            "intentional_drift: "
            f"missing={','.join(sorted(missing_state)) or 'none'}; "
            f"drift={','.join(sorted(drifted)) or 'none'}"
        )
    return "proven", "state+equivalence witnesses satisfied"


def _load_shim_lifecycle_rows(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    by_capability: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        capability_id = str(row.get("capability_id", "") or "").strip()
        if not capability_id:
            continue
        by_capability[capability_id] = {str(key): row[key] for key in row}
    return by_capability


def _build_rows(
    *,
    root: Path,
    state_payload: Mapping[str, object],
    equivalence_payload: Mapping[str, object],
    ledger_payload: Mapping[str, object],
    delta_jsonl_rows: list[dict[str, object]],
    shim_rows_by_capability: Mapping[str, dict[str, object]],
    shim_debt_rows: list[ShimDebtRow],
) -> list[dict[str, object]]:
    state_surfaces = _state_semantic_surfaces(state_payload)
    equivalence_classes = _surface_classification_map(equivalence_payload)

    ledger_records = ledger_payload.get("records")
    normalized_ledger_records: list[dict[str, object]] = []
    if isinstance(ledger_records, list):
        for record in ledger_records:
            if isinstance(record, Mapping):
                normalized_ledger_records.append({str(key): record[key] for key in record})
    if not normalized_ledger_records and delta_jsonl_rows:
        normalized_ledger_records = delta_jsonl_rows

    delta_surface_counts = _surface_delta_event_counts(
        ledger_records=normalized_ledger_records,
    )

    rows: list[dict[str, object]] = []
    for spec in _CAPABILITY_SPECS:
        required = list(spec.required_surfaces)
        state_present = [surface for surface in required if surface in state_surfaces]
        equivalence = {surface: equivalence_classes.get(surface, "") for surface in required}
        base_status, base_status_reason = _status_for_capability(
            spec=spec,
            state_surfaces=state_surfaces,
            equivalence_classes=equivalence_classes,
        )
        runtime_import_count = _runtime_import_count(
            root=root,
            module_path=spec.replacement_module,
        )
        transitive_runtime_import_count = _transitive_runtime_import_count(
            root=root,
            module_path=spec.replacement_module,
        )
        runtime_import_free = runtime_import_count == 0
        transitive_runtime_import_free = transitive_runtime_import_count == 0
        shim_metadata = _shim_row_for_spec(
            spec=spec,
            shim_rows_by_capability=shim_rows_by_capability,
            shim_debt_rows=shim_debt_rows,
        )
        if base_status == "blocked":
            status = "blocked"
            status_reason = base_status_reason
        elif runtime_import_free and transitive_runtime_import_free:
            status = base_status
            status_reason = base_status_reason
        else:
            status = "intentional_drift"
            reason_suffix = (
                "shim_metadata_present"
                if shim_metadata is not None
                else "missing_open_shim_metadata"
            )
            status_reason = (
                f"{base_status_reason}; runtime_import_count={runtime_import_count}; "
                f"transitive_runtime_import_count={transitive_runtime_import_count}; "
                f"{reason_suffix}"
            )
        rows.append(
            {
                "capability_id": spec.capability_id,
                "domain_id": spec.domain_id,
                "legacy_surface": spec.legacy_surface,
                "replacement_surface": spec.replacement_surface,
                "replacement_module": spec.replacement_module,
                "proof_kind": spec.proof_kind,
                "capability": spec.capability,
                "substitute": spec.substitute,
                "required_surfaces": required,
                "commutation_witness": {
                    "input_fixture": "identical_input_paths+flags",
                    "canonicalizer": (
                        "semantic_surface_presence+equivalence_classification+"
                        "delta_surface_event_counts+replacement_module_runtime_import_audit+"
                        "replacement_module_transitive_runtime_import_audit"
                    ),
                    "equivalence_predicate": "required_surfaces_present_and_non_drift",
                    "artifact_path": "aspf_state+aspf_equivalence+aspf_delta_ledger carriers",
                    "state_surface_presence": {
                        surface: (surface in state_surfaces) for surface in required
                    },
                    "state_surface_count": len(state_present),
                    "equivalence_classification": equivalence,
                    "delta_surface_event_count": {
                        surface: int(delta_surface_counts.get(surface, 0))
                        for surface in required
                    },
                    "replacement_module_runtime_import_count": runtime_import_count,
                    "replacement_module_runtime_import_free": runtime_import_free,
                    "replacement_module_transitive_runtime_import_count": (
                        transitive_runtime_import_count
                    ),
                    "replacement_module_transitive_runtime_import_free": (
                        transitive_runtime_import_free
                    ),
                },
                "replacement_module_runtime_import_count": runtime_import_count,
                "replacement_module_runtime_import_free": runtime_import_free,
                "replacement_module_transitive_runtime_import_count": (
                    transitive_runtime_import_count
                ),
                "replacement_module_transitive_runtime_import_free": (
                    transitive_runtime_import_free
                ),
                "shim_metadata": shim_metadata,
                "status": status,
                "status_reason": status_reason,
            }
        )

    return sorted(rows, key=lambda row: str(row["capability_id"]))


def _render_markdown(
    *,
    payload: Mapping[str, object],
) -> str:
    rows = payload.get("rows", [])
    row_count = len(rows) if isinstance(rows, list) else 0
    status_counts = payload.get("summary", {}).get("status_counts", {}) if isinstance(payload.get("summary"), Mapping) else {}
    domain_status = payload.get("summary", {}).get("domain_status", {}) if isinstance(payload.get("summary"), Mapping) else {}
    lines: list[str] = [
        "---",
        "doc_revision: 1",
        "reader_reintern: \"Reader-only: re-intern if doc_revision changed since you last read this doc.\"",
        "doc_id: legacy_dataflow_monolith_retirement_ledger",
        "doc_role: audit",
        "doc_scope:",
        "  - repo",
        "  - analysis",
        "  - aspf",
        "doc_authority: informative",
        "doc_requires:",
        "  - POLICY_SEED.md#policy_seed",
        "  - AGENTS.md#agent_obligations",
        "  - glossary.md#contract",
        "doc_reviewed_as_of:",
        "  POLICY_SEED.md#policy_seed: 2",
        "  AGENTS.md#agent_obligations: 2",
        "  glossary.md#contract: 1",
        "doc_review_notes:",
        "  POLICY_SEED.md#policy_seed: \"Ledger rows are projected from existing ASPF carriers; no bespoke proof substrate introduced.\"",
        "  AGENTS.md#agent_obligations: \"Projection output remains evidence-only and does not bypass correction-unit validation obligations.\"",
        "  glossary.md#contract: \"Capability rows expose commutation witness status as proven/intentional_drift/blocked.\"",
        "doc_change_protocol: \"POLICY_SEED.md#change_protocol\"",
        "doc_erasure:",
        "  - formatting",
        "  - typos",
        "doc_owner: maintainer",
        "---",
        "",
        "<a id=\"legacy_dataflow_monolith_retirement_ledger\"></a>",
        "# Dataflow Runtime Retirement Ledger",
        "",
        "Projected from ASPF carriers (`trace`, `equivalence`, `state`, `delta_ledger`).",
        "",
        "Deterministic probe contract:",
        f"- state snapshot: `{_PROBE_STATE_SNAPSHOT}`",
        f"- delta jsonl: `{_PROBE_DELTA_JSONL}`",
        "- required surfaces: "
        + ", ".join(f"`{surface}`" for surface in _PROBE_REQUIRED_SURFACES),
        "",
        f"Rows: `{row_count}`",
        (
            "Status counts: "
            f"proven={status_counts.get('proven', 0)}, "
            f"intentional_drift={status_counts.get('intentional_drift', 0)}, "
            f"blocked={status_counts.get('blocked', 0)}"
        ),
        (
            "Domain status: "
            + ", ".join(
                f"{domain}={status}"
                for domain, status in (
                    sorted(domain_status.items())
                    if isinstance(domain_status, Mapping)
                    else []
                )
            )
        ),
        "",
        "| Capability ID | Domain | Proof kind | Replacement module | Required surfaces | Status | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            required = row.get("required_surfaces", [])
            required_text = ", ".join(required) if isinstance(required, list) else ""
            capability_id = str(row.get("capability_id", ""))
            domain_id = str(row.get("domain_id", ""))
            proof_kind = str(row.get("proof_kind", ""))
            replacement_module = str(row.get("replacement_module", ""))
            status = str(row.get("status", ""))
            reason = str(row.get("status_reason", ""))
            lines.append(
                f"| `{capability_id}` | `{domain_id}` | `{proof_kind}` | `{replacement_module}` | `{required_text}` | `{status}` | `{reason}` |"
            )

    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _status_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {"proven": 0, "intentional_drift": 0, "blocked": 0}
    for row in rows:
        status = str(row.get("status", "") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _domain_status(rows: list[dict[str, object]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        domain_id = str(row.get("domain_id", "") or "").strip()
        status = str(row.get("status", "") or "").strip()
        if not domain_id or not status:
            continue
        grouped.setdefault(domain_id, []).append(status)
    status_by_domain: dict[str, str] = {}
    for domain_id, statuses in grouped.items():
        if all(item == "proven" for item in statuses):
            status_by_domain[domain_id] = "proven"
        elif any(item == "blocked" for item in statuses):
            status_by_domain[domain_id] = "blocked"
        else:
            status_by_domain[domain_id] = "intentional_drift"
    return dict(sorted(status_by_domain.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Project the dataflow-runtime retirement ledger from existing ASPF "
            "trace/equivalence/state/delta carriers."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--aspf-trace", default="", help="ASPF trace JSON path")
    parser.add_argument("--aspf-equivalence", default="", help="ASPF equivalence JSON path")
    parser.add_argument(
        "--aspf-state",
        default=_PROBE_STATE_SNAPSHOT,
        help="ASPF state JSON path",
    )
    parser.add_argument(
        "--aspf-delta-ledger",
        default="",
        help="ASPF delta ledger JSON path (defaults to state.delta_ledger)",
    )
    parser.add_argument(
        "--aspf-delta-jsonl",
        default=_PROBE_DELTA_JSONL,
        help="ASPF delta JSONL path",
    )
    parser.add_argument(
        "--shim-lifecycle-json",
        default="",
        help="Optional shim lifecycle JSON payload with rows[].capability_id",
    )
    parser.add_argument(
        "--shim-lifecycle-md",
        default="docs/compatibility_layer_debt_register.md",
        help="Optional shim lifecycle markdown register path",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/audit_reports/legacy_dataflow_monolith_retirement_ledger.json",
        help="Output JSON ledger path",
    )
    parser.add_argument(
        "--output-md",
        default="docs/audits/legacy_dataflow_monolith_retirement_ledger.md",
        help="Output markdown ledger path",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    trace_path = (root / args.aspf_trace).resolve() if args.aspf_trace else None
    equivalence_path = (root / args.aspf_equivalence).resolve() if args.aspf_equivalence else None
    state_path = (root / args.aspf_state).resolve() if args.aspf_state else None
    delta_ledger_path = (
        (root / args.aspf_delta_ledger).resolve() if args.aspf_delta_ledger else None
    )
    delta_jsonl_path = (root / args.aspf_delta_jsonl).resolve() if args.aspf_delta_jsonl else None
    shim_path = (root / args.shim_lifecycle_json).resolve() if args.shim_lifecycle_json else None
    shim_md_path = (root / args.shim_lifecycle_md).resolve() if args.shim_lifecycle_md else None

    trace_payload = _load_json(trace_path)
    equivalence_payload = _load_json(equivalence_path)
    state_payload = _load_json(state_path)
    if not trace_payload:
        state_trace = state_payload.get("trace")
        if isinstance(state_trace, Mapping):
            trace_payload = {str(key): state_trace[key] for key in state_trace}
    if not equivalence_payload:
        state_equivalence = state_payload.get("equivalence")
        if isinstance(state_equivalence, Mapping):
            equivalence_payload = {
                str(key): state_equivalence[key] for key in state_equivalence
            }
    _validate_required_witness_surfaces(
        state_payload=state_payload,
        equivalence_payload=equivalence_payload,
    )
    ledger_payload = _load_json(delta_ledger_path)
    if not ledger_payload:
        ledger_from_state = state_payload.get("delta_ledger")
        if isinstance(ledger_from_state, Mapping):
            ledger_payload = {str(key): ledger_from_state[key] for key in ledger_from_state}
    delta_jsonl_rows = _load_jsonl(delta_jsonl_path)
    shim_rows_by_capability = _load_shim_lifecycle_rows(shim_path)
    shim_debt_rows = _load_shim_lifecycle_rows_from_md(shim_md_path)

    rows = _build_rows(
        root=root,
        state_payload=state_payload,
        equivalence_payload=equivalence_payload,
        ledger_payload=ledger_payload,
        delta_jsonl_rows=delta_jsonl_rows,
        shim_rows_by_capability=shim_rows_by_capability,
        shim_debt_rows=shim_debt_rows,
    )

    inputs = {
        "trace": trace_payload,
        "equivalence": equivalence_payload,
        "state": state_payload,
        "delta_ledger": ledger_payload,
        "delta_jsonl_row_count": len(delta_jsonl_rows),
    }
    projection_digest = _digest_payload(payloads=inputs)

    payload: dict[str, object] = {
        "format_version": 1,
        "projection_digest": projection_digest,
        "inputs": {
            "aspf_trace": str(trace_path) if trace_path is not None else None,
            "aspf_equivalence": str(equivalence_path) if equivalence_path is not None else None,
            "aspf_state": str(state_path) if state_path is not None else None,
            "aspf_delta_ledger": str(delta_ledger_path) if delta_ledger_path is not None else None,
            "aspf_delta_jsonl": str(delta_jsonl_path) if delta_jsonl_path is not None else None,
            "shim_lifecycle_json": str(shim_path) if shim_path is not None else None,
            "shim_lifecycle_md": str(shim_md_path) if shim_md_path is not None else None,
        },
        "summary": {
            "status_counts": _status_counts(rows),
            "domain_status": _domain_status(rows),
            "row_count": len(rows),
        },
        "rows": rows,
    }

    json_path = (root / args.output_json).resolve()
    md_path = (root / args.output_md).resolve()
    _write_json(json_path, payload)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_render_markdown(payload=payload), encoding="utf-8")

    print(f"wrote retirement ledger JSON: {json_path}")
    print(f"wrote retirement ledger markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
