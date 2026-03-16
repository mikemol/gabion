#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import cache
import json
import os
import re
import sys
import tomllib
import subprocess
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Mapping, Tuple, TypeAlias, cast

from gabion.frontmatter_ingress import (
    FrontmatterParseMode,
    parse_frontmatter_document,
    scan_frontmatter_lines,
)
from gabion.tooling.runtime.deadline_runtime import DeadlineBudget, deadline_scope_from_ticks
from gabion.analysis.aspf.aspf import Forest
from gabion.analysis.foundation.timeout_context import check_deadline
from gabion.analysis.projection.projection_exec import apply_execution_ops
from gabion.analysis.projection.projection_exec_plan import execution_ops_from_spec
from gabion.analysis.projection.projection_normalize import normalize_spec, spec_canonical_json, spec_hash
from gabion.analysis.projection.projection_spec import ProjectionOp, ProjectionSpec, spec_from_dict
from gabion.analysis.semantics import evidence_keys
from gabion.analysis.semantics.impact_index import build_impact_index
from gabion.governance_paths import GOVERNANCE_PATHS
from gabion.analysis.semantics.obligation_registry import (
    evaluate_obligations, summarize_obligations)
from gabion.invariants import decision_protocol, never
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.governance.governance_rules import load_governance_rules
from gabion_governance.compliance_render import render_compliance
from gabion_governance.compliance_render.decision_contracts import (
    ConsolidationConfig,
    DecisionSurface,
    LintEntry,
)
from gabion_governance.consolidation_audit import parse_lint_entry as _parse_lint_entry
from gabion_governance.consolidation_audit import parse_surface_line
from gabion_governance.docflow_audit import (
    AgentDirective,
    Doc,
    DocflowAuditContext,
    DocflowInvariant,
    DocflowObligationResult,
    DocflowPredicateMatcher,
    run_docflow_domain,
)
from gabion_governance.docflow_audit.contracts import (
    Frontmatter,
    FrontmatterValue,
    JSONValue,
)
from gabion_governance.governance_audit_contracts import GovernanceAuditAggregateResult
from gabion_governance.sppf_audit import build_sppf_graph, run_status_consistency
from gabion_governance.sppf_audit.contracts import SppfStatusConsistencyResult

_DEFAULT_AUDIT_TIMEOUT_TICKS = 120_000
_DEFAULT_AUDIT_TIMEOUT_TICK_NS = 1_000_000
_DEFAULT_AUDIT_TIMEOUT_BUDGET = DeadlineBudget(
    ticks=_DEFAULT_AUDIT_TIMEOUT_TICKS,
    tick_ns=_DEFAULT_AUDIT_TIMEOUT_TICK_NS,
)
_DEFAULT_AUDIT_GAS_LIMIT = 50_000_000
_AUDIT_GAS_LIMIT_ENV = "GABION_AUDIT_GAS_LIMIT"


def _audit_gas_limit() -> int:
    raw = os.getenv(_AUDIT_GAS_LIMIT_ENV, "").strip()
    if not raw:
        return _DEFAULT_AUDIT_GAS_LIMIT
    try:
        value = int(raw)
    except ValueError:
        never("invalid audit gas limit", value=raw)
    if value <= 0:
        never("invalid audit gas limit", value=value)
    return value


def _audit_deadline_scope():
    return deadline_scope_from_ticks(
        budget=_DEFAULT_AUDIT_TIMEOUT_BUDGET,
        gas_limit=_audit_gas_limit(),
    )


def _sorted(values: Iterable[object], *, key: Callable[[object], object] | None = None, reverse: bool = False) -> list[object]:
    return ordered_or_sorted(
        values,
        source="gabion.tooling.governance_audit",
        key=key,
        reverse=reverse,
    )


# --- Docflow audit constants ---

CORE_GOVERNANCE_DOCS = [
    "POLICY_SEED.md",
    "glossary.md",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
]

GOVERNANCE_DOCS = CORE_GOVERNANCE_DOCS + [
    "docs/governance_control_loops.md",
    "docs/governance_loop_matrix.md",
    "docs/publishing_practices.md",
    "docs/influence_index.md",
    "docs/coverage_semantics.md",
    "docs/normative_clause_index.md",
    "docs/matrix_acceptance.md",
    "docs/sppf_checklist.md",
]

_REVIEW_NOTE_REVISION_LINT_DOCS = frozenset(
    {
        "AGENTS.md",
        "README.md",
        "CONTRIBUTING.md",
        "POLICY_SEED.md",
        "glossary.md",
        "docs/normative_clause_index.md",
    }
)

GOVERNANCE_CONTROL_LOOPS_DOC = "docs/governance_control_loops.md"

NORMATIVE_LOOP_DOMAINS = (
    "security/workflows",
    "docs/docflow",
    "LSP architecture",
    "dataflow grammar",
    "baseline ratchets",
)


def _iter_in_governance_relpaths(root: Path) -> list[str]:
    in_root = root / "in"
    if not in_root.exists():
        return []
    relpaths: list[str] = []
    for path in _sorted(in_root.glob("in-*.md")):
        check_deadline()
        if not isinstance(path, Path):
            continue
        relpaths.append(path.relative_to(root).as_posix())
    return relpaths


def _iter_out_governance_relpaths(root: Path) -> list[str]:
    out_root = root / "out"
    if not out_root.exists():
        return []
    relpaths: list[str] = []
    for path in _sorted(out_root.glob("out-*.md")):
        check_deadline()
        if not isinstance(path, Path):
            continue
        relpaths.append(path.relative_to(root).as_posix())
    return relpaths


def _iter_default_docflow_relpaths(root: Path) -> list[str]:
    relpaths: list[str] = []
    seen: set[str] = set()
    for rel in [*GOVERNANCE_DOCS, *_iter_in_governance_relpaths(root)]:
        check_deadline()
        if rel in seen:
            continue
        seen.add(rel)
        relpaths.append(rel)
    return relpaths

REQUIRED_FIELDS = [
    "doc_id",
    "doc_role",
    "doc_scope",
    "doc_authority",
    "doc_requires",
    "doc_reviewed_as_of",
    "doc_review_notes",
    "doc_change_protocol",
]
LIST_FIELDS = {
    "doc_scope",
    "doc_requires",
    "doc_commutes_with",
    "doc_invariants",
    "doc_erasure",
}
MAP_FIELDS = {
    "doc_reviewed_as_of",
    "doc_review_notes",
    "doc_sections",
    "doc_section_requires",
    "doc_section_reviews",
}

FOREST_FALLBACK_MARKER = "FOREST_FALLBACK_USED"
FOREST_FALLBACK_WARNING_CLASS = "consolidation.forest_fallback"
CONSOLIDATION_SOURCE_FOREST_NATIVE = "forest-native"
CONSOLIDATION_SOURCE_FALLBACK_DERIVED = "fallback-derived"


def _coerce_argv(argv: list[str] | None) -> list[str]:
    return argv if argv is not None else sys.argv[1:]


def _latest_snapshot_dir(root: Path) -> Path:
    marker = root / "artifacts" / "audit_snapshots" / "LATEST.txt"
    if not marker.exists():
        raise FileNotFoundError(marker)
    stamp = marker.read_text().strip()
    if not stamp:
        raise ValueError("LATEST.txt is empty")
    return root / "artifacts" / "audit_snapshots" / stamp


def _scope_match(path: str, scope: str | None) -> bool:
    if scope is None:
        return True
    scope_name = Path(scope).name
    return path == scope or path == scope_name or path.endswith(scope) or path.endswith(scope_name)


def _latest_lint_path(root: Path) -> Path:
    snapshot = _latest_snapshot_dir(root)
    return snapshot / "lint.txt"


def _load_consolidation_config(root: Path) -> ConsolidationConfig:
    config_path = root / "gabion.toml"
    if not config_path.exists():
        return ConsolidationConfig()
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ConsolidationConfig()
    section = data.get("consolidation", {})
    if not isinstance(section, dict):
        return ConsolidationConfig()

    def _coerce_int(key: str, default: int) -> int:
        try:
            return int(section.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _coerce_bool(key: str, default: bool) -> bool:
        value = section.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    return ConsolidationConfig(
        min_functions=_coerce_int("min_functions", 3),
        min_files=_coerce_int("min_files", 2),
        max_examples=_coerce_int("max_examples", 5),
        require_forest=_coerce_bool("require_forest", True),
    )


# --- Docflow audit helpers ---

SPPF_TAG_RE = re.compile(r"sppf\{([^}]*)\}")
SPPF_LINE_RE = re.compile(r"^- \[(?P<state>[x~ ])\]\s+(?P<body>.*)$")
SPPF_IN_REF_RE = re.compile(r"\((?:in/)?in-\d+")
SPPF_DOC_REF_SPLIT_RE = re.compile(r"\s*,\s*")
SPPF_ALLOWED_STATUSES = {"done", "partial", "planned", "blocked", "deprecated"}
DOCFLOW_NOTION_RE = re.compile(r"\b(must\s+not|shall\s+not|do\s+not|never|must|shall|required)\b", re.IGNORECASE)
INFLUENCE_STATUS_MAP = {
    "adopted": "done",
    "partial": "partial",
    "queued": "planned",
    "untriaged": "planned",
    "rejected": "blocked",
}
STATUS_TRIPLET_OVERRIDE_MARKER = "docflow:status-triplet-override"
CHECKLIST_STATE_MAP = {"x": "done", "~": "partial", " ": "planned"}
DECLARATION_TO_INFLUENCE_MAP = {
    "planned": {"queued", "untriaged"},
    "queued": {"queued"},
    "partial": {"partial"},
    "adopted": {"adopted"},
    "rejected": {"rejected"},
}
DECLARATION_TO_CHECKLIST_MAP = {
    "planned": {"planned"},
    "queued": {"planned"},
    "partial": {"partial"},
    "adopted": {"done"},
    "rejected": {"blocked"},
}


def _matrix_gate_ids_from_markdown(body: str) -> set[str]:
    gate_ids: set[str] = set()
    for line in body.splitlines():
        check_deadline()
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        gate_cell = cells[1].strip("`").strip()
        if not gate_cell or gate_cell == "gate ID" or gate_cell.startswith("---"):
            continue
        gate_ids.add(gate_cell)
    return gate_ids


def _parse_sppf_tag(payload: str) -> dict[str, str]:
    items: dict[str, str] = {}
    for chunk in payload.split(";"):
        check_deadline()
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        items[key.strip()] = value.strip()
    return items


def _parse_doc_ref(value: str) -> list[tuple[str, int | None]]:
    refs: list[tuple[str, int | None]] = []
    for part in SPPF_DOC_REF_SPLIT_RE.split(value):
        check_deadline()
        chunk = part.strip()
        if not chunk:
            continue
        if chunk.startswith("in/"):
            chunk = chunk[3:]
        name, _, rev = chunk.partition("@")
        name = name.strip()
        if not name:
            continue
        if name.endswith(".md"):
            name = name[:-3]
        if name.startswith("in-") and name[3:].isdigit():
            doc_id = name
        elif name.startswith("in-") and name[3:].split("/")[0].isdigit():
            doc_id = name.split("/", 1)[0]
        elif name.startswith("docs/") or "/" in name or name.endswith(".md"):
            doc_id = name if name.endswith(".md") else f"{name}.md"
        else:
            doc_id = name
        rev_value: int | None = None
        if rev:
            try:
                rev_value = int(rev)
            except ValueError:
                rev_value = None
        refs.append((doc_id, rev_value))
    return refs


def _format_doc_ref(doc_id: str, rev: int | None) -> str:
    return f"{doc_id}@{rev}" if rev is not None else doc_id


_DOCFLOW_PREDICATE_NAMES = frozenset(
    {
        "missing_frontmatter",
        "missing_required_field",
        "invalid_field_type",
        "missing_governance_ref",
        "missing_explicit_ref",
        "review_pin_mismatch",
        "missing_review_note",
        "review_note_revision_mismatch",
        "commutation_unreciprocated",
        "evidence_row",
        "evidence_kind",
        "evidence_id",
        "evidence_source",
        "missing_loop_entry",
        "missing_matrix_gate_entry",
        "implication_matrix_conflict",
    }
)


def _normalize_docflow_predicates(predicates: Iterable[str]) -> tuple[str, ...]:
    cleaned: dict[str, None] = {}
    for predicate in predicates:
        normalized = str(predicate).strip()
        if not normalized:
            continue
        cleaned.setdefault(normalized, None)
    return tuple(str(value) for value in _sorted(cleaned))


def _copy_docflow_matcher_params(
    params: Mapping[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return {
        str(key): value
        for key, value in (params or {}).items()
    }


def _parse_docflow_predicate_matcher(
    *,
    predicates: Iterable[str],
    params: Mapping[str, JSONValue] | None = None,
) -> DocflowPredicateMatcher | None:
    normalized = _normalize_docflow_predicates(predicates)
    if not normalized:
        return None
    if any(predicate not in _DOCFLOW_PREDICATE_NAMES for predicate in normalized):
        return None
    return DocflowPredicateMatcher(
        predicates=normalized,
        params=_copy_docflow_matcher_params(params),
    )


def _make_invariant_matcher(
    name: str,
    predicates: Iterable[str],
    *,
    params: Mapping[str, JSONValue] | None = None,
) -> DocflowPredicateMatcher:
    matcher = _parse_docflow_predicate_matcher(
        predicates=predicates,
        params=params,
    )
    if matcher is None:
        never(
            "invalid internal docflow invariant matcher",
            name=name,
            predicates=list(predicates),
        )
    return matcher


def _matcher_from_spec(spec: ProjectionSpec) -> DocflowPredicateMatcher | None:
    payload = normalize_spec(spec)
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, list):
        return None
    predicates: list[str] = []
    for entry in pipeline:
        check_deadline()
        if not isinstance(entry, dict):
            return None
        if str(entry.get("op", "") or "") != "select":
            return None
        raw_params = entry.get("params", {})
        if not isinstance(raw_params, dict):
            return None
        params = {str(key): raw_params[key] for key in raw_params}
        raw_predicates = params.get("predicates", [])
        match raw_predicates:
            case str() as predicate if predicate.strip():
                predicates.append(predicate.strip())
            case list() as predicate_list:
                for raw_value in predicate_list:
                    check_deadline()
                    text = str(raw_value).strip()
                    if text:
                        predicates.append(text)
            case _:
                return None
    raw_matcher_params = payload.get("params", {})
    if not isinstance(raw_matcher_params, dict):
        return None
    return _parse_docflow_predicate_matcher(
        predicates=predicates,
        params={str(key): raw_matcher_params[key] for key in raw_matcher_params},
    )


def _extract_doc_body_notions_with_anchors(*, path: str, doc: Doc) -> list[dict[str, JSONValue]]:
    notions: list[dict[str, JSONValue]] = []
    lines = doc.body.splitlines()
    for line_index, raw_line in enumerate(lines, start=1):
        check_deadline()
        line = raw_line.strip()
        if not line:
            continue
        for match in DOCFLOW_NOTION_RE.finditer(raw_line):
            check_deadline()
            token = match.group(1).strip().lower()
            normalized_token = " ".join(token.split())
            polarity = "negative" if "not" in normalized_token or normalized_token == "never" else "positive"
            subject = re.sub(r"\s+", " ", raw_line[match.end() :].strip(" :.-`*_[]()\t")).lower()
            if not subject:
                continue
            notions.append(
                {
                    "path": path,
                    "qual": str(doc.frontmatter.get("doc_id") or path),
                    "row_kind": "doc_body_notion",
                    "token": normalized_token,
                    "polarity": polarity,
                    "subject": subject,
                    "anchor": {
                        "path": path,
                        "line": line_index,
                        "column": match.start() + 1,
                    },
                }
            )
    return sorted(notions, key=lambda item: (str(item["path"]), int(item["anchor"]["line"]), int(item["anchor"]["column"])))


def _build_doc_implication_lattice(
    *,
    path: str,
    notions: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    indexed_nodes: list[dict[str, JSONValue]] = []
    for idx, notion in enumerate(notions):
        check_deadline()
        indexed_nodes.append({"index": idx, **notion})
    n = len(indexed_nodes)
    second_order_matrix = [[0 for _ in range(n)] for _ in range(n)]
    second_order_edges: list[dict[str, JSONValue]] = []
    subject_to_indices: dict[str, list[int]] = defaultdict(list)
    for node in indexed_nodes:
        check_deadline()
        subject = str(node.get("subject") or "")
        if not subject:
            continue
        subject_to_indices[subject].append(int(node["index"]))
    for subject, indices in subject_to_indices.items():
        check_deadline()
        for left_offset, left_index in enumerate(indices):
            left = indexed_nodes[left_index]
            for right_offset, right_index in enumerate(indices[left_offset + 1 :], start=1):
                if (right_offset & 63) == 0:
                    check_deadline()
                right = indexed_nodes[right_index]
                relation = (
                    "reinforces"
                    if left.get("polarity") == right.get("polarity")
                    else "conflicts"
                )
                second_order_matrix[left_index][right_index] = (
                    1 if relation == "reinforces" else -1
                )
                second_order_edges.append(
                    {
                        "relation": relation,
                        "left": left_index,
                        "right": right_index,
                        "subject": subject,
                    }
                )
    return {
        "path": path,
        "first_order": {"notions": indexed_nodes},
        "second_order": {"matrix": second_order_matrix, "implications": second_order_edges},
        # Third-order expansion is intentionally omitted to keep repo-scale
        # docflow runs bounded; first/second-order matrices remain canonical.
        "third_order": {"chains": []},
    }


def _compose_doc_dependency_matrices(
    *,
    docs: dict[str, Doc],
    per_doc_lattices: dict[str, dict[str, JSONValue]],
) -> tuple[dict[str, dict[str, JSONValue]], list[str], list[dict[str, object]]]:
    composed: dict[str, dict[str, JSONValue]] = {}
    warnings: list[str] = []
    conflict_rows: list[dict[str, object]] = []

    def _deps_for(rel: str) -> list[str]:
        fm = docs[rel].frontmatter
        requires = fm.get("doc_requires", [])
        if not isinstance(requires, list):
            return []
        deps: list[str] = []
        for item in requires:
            check_deadline()
            if not isinstance(item, str):
                continue
            base = str(_doc_ref_base(item))
            if not base.endswith(".md"):
                continue
            deps.append(base)
        return deps

    def _notion_identity(notion: dict[str, JSONValue]) -> tuple[object, ...]:
        anchor = notion.get("anchor")
        line = 0
        column = 0
        if isinstance(anchor, Mapping):
            line = int(anchor.get("line", 0) or 0)
            column = int(anchor.get("column", 0) or 0)
        return (
            str(notion.get("path", "") or ""),
            line,
            column,
            str(notion.get("subject", "") or ""),
            str(notion.get("polarity", "") or ""),
        )

    memo: dict[str, list[dict[str, JSONValue]]] = {}

    def _collect_notions(rel: str, chain: set[str]) -> list[dict[str, JSONValue]]:
        if rel in memo:
            return memo[rel]
        if rel in chain:
            return list(per_doc_lattices.get(rel, {}).get("first_order", {}).get("notions", []))
        chain_next = set(chain)
        chain_next.add(rel)
        merged: list[dict[str, JSONValue]] = list(per_doc_lattices.get(rel, {}).get("first_order", {}).get("notions", []))
        for dep in _deps_for(rel):
            check_deadline()
            if dep not in docs:
                warnings.append(f"{rel}: dependency matrix source missing for {dep}")
                continue
            merged.extend(_collect_notions(dep, chain_next))
        unique_notions: dict[tuple[object, ...], dict[str, JSONValue]] = {}
        for notion in merged:
            check_deadline()
            identity = _notion_identity(notion)
            if identity not in unique_notions:
                unique_notions[identity] = notion
        merged = list(unique_notions.values())
        merged = sorted(
            merged,
            key=lambda item: (
                str(item.get("path", "")),
                int(item.get("anchor", {}).get("line", 0)),
                int(item.get("anchor", {}).get("column", 0)),
            ),
        )
        memo[rel] = merged
        return merged

    for rel in _sorted(docs.keys()):
        check_deadline()
        notions = _collect_notions(rel, set())
        lattice = _build_doc_implication_lattice(path=rel, notions=notions)
        composed[rel] = lattice
        by_subject: dict[str, set[str]] = defaultdict(set)
        by_subject_anchor: dict[tuple[str, str], dict[str, JSONValue]] = {}
        for node in lattice["first_order"]["notions"]:
            check_deadline()
            subject = str(node.get("subject") or "")
            polarity = str(node.get("polarity") or "")
            if not subject or not polarity:
                continue
            by_subject[subject].add(polarity)
            by_subject_anchor[(subject, polarity)] = node
        for subject, polarities in by_subject.items():
            check_deadline()
            if len(polarities) < 2:
                continue
            left = by_subject_anchor.get((subject, "positive"), {})
            right = by_subject_anchor.get((subject, "negative"), {})
            conflict_rows.append(
                {
                    "row_kind": "doc_implication_matrix_conflict",
                    "path": rel,
                    "qual": str(docs[rel].frontmatter.get("doc_id") or rel),
                    "subject": subject,
                    "left_anchor": left.get("anchor"),
                    "right_anchor": right.get("anchor"),
                    "scope": "dependency_composed",
                }
            )
    return composed, warnings, conflict_rows


def _docflow_predicates() -> dict[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]]:
    def _is_row(row: Mapping[str, JSONValue], kind: str) -> bool:
        return str(row.get("row_kind", "") or "") == kind

    def _param_list(params: Mapping[str, JSONValue], *keys: str) -> list[str]:
        for key in keys:
            check_deadline()
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
        return []

    def _missing_frontmatter(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_missing_frontmatter")

    def _missing_required_field(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_required_field") and not bool(row.get("present", False))

    def _invalid_field_type(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_field_type") and not bool(row.get("valid", False))

    def _missing_governance_ref(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_missing_governance_ref")

    def _missing_explicit_ref(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_requires_ref") and not bool(row.get("explicit", False))

    def _review_pin_mismatch(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_review_pin") and not bool(row.get("match", False))

    def _missing_review_note(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_review_note") and not bool(row.get("note_present", False))

    def _review_note_revision_mismatch(
        row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]
    ) -> bool:
        return _is_row(row, "doc_review_note_revision") and not bool(row.get("match", False))

    def _commute_unreciprocated(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_commute_edge") and not bool(row.get("reciprocated", False))

    def _evidence_row(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "evidence_key")

    def _evidence_kind(row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]) -> bool:
        if not _is_row(row, "evidence_key"):
            return False
        kinds = _param_list(params, "evidence_kind", "evidence_kinds")
        if not kinds:
            return False
        return str(row.get("evidence_kind", "") or "") in kinds

    def _evidence_id(row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]) -> bool:
        if not _is_row(row, "evidence_key"):
            return False
        ids = _param_list(params, "evidence_id", "evidence_ids")
        if not ids:
            return False
        return str(row.get("evidence_id", "") or "") in ids

    def _evidence_source(row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]) -> bool:
        if not _is_row(row, "evidence_key"):
            return False
        sources = _param_list(params, "evidence_source", "evidence_sources")
        if not sources:
            return False
        return str(row.get("evidence_source", "") or "") in sources

    def _missing_loop_entry(row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]) -> bool:
        if not _is_row(row, "doc_loop_entry"):
            return False
        if row.get("required") is not True:
            return False
        return row.get("declared") is not True

    def _missing_matrix_gate_entry(row: Mapping[str, JSONValue], params: Mapping[str, JSONValue]) -> bool:
        if not _is_row(row, "doc_loop_matrix_gate"):
            return False
        if row.get("required") is not True:
            return False
        return row.get("declared") is not True

    def _implication_matrix_conflict(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        return _is_row(row, "doc_implication_matrix_conflict")

    return {
        "missing_frontmatter": _missing_frontmatter,
        "missing_required_field": _missing_required_field,
        "invalid_field_type": _invalid_field_type,
        "missing_governance_ref": _missing_governance_ref,
        "missing_explicit_ref": _missing_explicit_ref,
        "review_pin_mismatch": _review_pin_mismatch,
        "missing_review_note": _missing_review_note,
        "review_note_revision_mismatch": _review_note_revision_mismatch,
        "commutation_unreciprocated": _commute_unreciprocated,
        "evidence_row": _evidence_row,
        "evidence_kind": _evidence_kind,
        "evidence_id": _evidence_id,
        "evidence_source": _evidence_source,
        "missing_loop_entry": _missing_loop_entry,
        "missing_matrix_gate_entry": _missing_matrix_gate_entry,
        "implication_matrix_conflict": _implication_matrix_conflict,
    }


_DOCFLOW_EVIDENCE_PREDICATES = frozenset(
    {
        "evidence_row",
        "evidence_kind",
        "evidence_id",
        "evidence_source",
    }
)


def _invariant_uses_evidence_rows(invariant: DocflowInvariant) -> bool:
    return any(
        predicate in _DOCFLOW_EVIDENCE_PREDICATES
        for predicate in invariant.matcher.predicates
    )


DOCFLOW_AUDIT_INVARIANTS = [
    DocflowInvariant(
        name="docflow:missing_frontmatter",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_frontmatter", ["missing_frontmatter"]),
    ),
    DocflowInvariant(
        name="docflow:missing_required_field",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_required_field", ["missing_required_field"]),
    ),
    DocflowInvariant(
        name="docflow:invalid_field_type",
        kind="never",
        matcher=_make_invariant_matcher("docflow:invalid_field_type", ["invalid_field_type"]),
    ),
    DocflowInvariant(
        name="docflow:missing_governance_ref",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_governance_ref", ["missing_governance_ref"]),
    ),
    DocflowInvariant(
        name="docflow:missing_explicit_reference",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_explicit_reference", ["missing_explicit_ref"]),
    ),
    DocflowInvariant(
        name="docflow:review_pin_mismatch",
        kind="never",
        matcher=_make_invariant_matcher("docflow:review_pin_mismatch", ["review_pin_mismatch"]),
    ),
    DocflowInvariant(
        name="docflow:missing_review_note",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_review_note", ["missing_review_note"]),
    ),
    DocflowInvariant(
        name="docflow:review_note_revision_mismatch",
        kind="never",
        matcher=_make_invariant_matcher(
            "docflow:review_note_revision_mismatch",
            ["review_note_revision_mismatch"],
        ),
    ),
    DocflowInvariant(
        name="docflow:commutation_unreciprocated",
        kind="never",
        matcher=_make_invariant_matcher("docflow:commutation_unreciprocated", ["commutation_unreciprocated"]),
    ),
    DocflowInvariant(
        name="docflow:missing_governance_control_loop",
        kind="never",
        matcher=_make_invariant_matcher("docflow:missing_governance_control_loop", ["missing_loop_entry"]),
    ),
    DocflowInvariant(
        name="docflow:governance_loop_matrix_drift",
        kind="never",
        matcher=_make_invariant_matcher("docflow:governance_loop_matrix_drift", ["missing_matrix_gate_entry"]),
    ),
    DocflowInvariant(
        name="docflow:implication_matrix_conflict",
        kind="never",
        matcher=_make_invariant_matcher("docflow:implication_matrix_conflict", ["implication_matrix_conflict"]),
    ),
]


def _load_issues_json(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    issues: dict[str, dict[str, object]] = {}
    for entry in payload:
        check_deadline()
        if not isinstance(entry, dict):
            continue
        number = entry.get("number")
        if not isinstance(number, int):
            continue
        key = f"GH-{number}"
        issues[key] = entry
    return issues


def _slugify_heading(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _frontmatter_block(lines: list[str]) -> tuple[list[str], int] | None:
    block = scan_frontmatter_lines(lines)
    if block is None:
        return None
    fm_lines, end = block
    return list(fm_lines), end


def _suite_key(
    *,
    domain: str,
    kind: str,
    path: str,
    qual: str,
    span: tuple[int, int, int, int],
) -> tuple[object, ...]:
    return (domain, kind, path, qual, *span)


def _add_suite_node(
    forest: Forest,
    *,
    domain: str,
    kind: str,
    path: str,
    qual: str,
    span: tuple[int, int, int, int],
    parent: tuple[object, ...] | None,
) -> tuple[object, ...]:
    key = _suite_key(domain=domain, kind=kind, path=path, qual=qual, span=span)
    meta: dict[str, object] = {
        "domain": domain,
        "kind": kind,
        "path": path,
        "qual": qual,
        "span": list(span),
    }
    if parent is not None:
        meta["parent"] = list(parent)
    node_id = forest.add_node("SuiteSite", key, meta=meta)
    if parent is not None:
        parent_node = forest.add_node("SuiteSite", parent, meta=None)
        forest.add_alt("SuiteContains", (parent_node, node_id))
    return key


def _iter_docflow_paths(root: Path, extra_paths: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for rel in _iter_default_docflow_relpaths(root):
        check_deadline()
        path = root / rel
        if path.exists():
            if path not in seen:
                paths.append(path)
                seen.add(path)
    if extra_paths:
        for entry in extra_paths:
            check_deadline()
            if not entry:
                continue
            raw = Path(entry)
            path = raw if raw.is_absolute() else root / raw
            if path.is_dir():
                for doc in _sorted(path.rglob("*.md")):
                    check_deadline()
                    if doc not in seen:
                        paths.append(doc)
                        seen.add(doc)
            elif path.is_file():
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
    return paths


def _suite_meta(
    *,
    key: tuple[object, ...],
    domain: str,
    kind: str,
    path: str,
    qual: str,
    span: tuple[int, int, int, int],
    parent: tuple[object, ...] | None,
) -> dict[str, object]:
    return {
        "suite_key": list(key),
        "suite_domain": domain,
        "suite_kind": kind,
        "path": path,
        "qual": qual,
        "span_line": span[0],
        "span_col": span[1],
        "span_end_line": span[2],
        "span_end_col": span[3],
        "parent_key": list(parent) if parent is not None else None,
    }


def _emit_docflow_suite_artifacts(
    *,
    root: Path,
    extra_paths: list[str] | None,
    issues_json: Path | None,
    forest_output: Path | None,
    relation_output: Path | None,
) -> None:
    doc_paths = _iter_docflow_paths(root, extra_paths)
    issues = _load_issues_json(issues_json)
    forest = Forest()
    rows: list[dict[str, object]] = []
    suite_meta_index: dict[tuple[object, ...], dict[str, object]] = {}
    docs: dict[str, Doc] = {}
    missing_frontmatter: set[str] = set()
    doc_suite_by_rel: dict[str, tuple[object, ...]] = {}

    def _record_suite(
        *,
        domain: str,
        kind: str,
        path: str,
        qual: str,
        span: tuple[int, int, int, int],
        parent: tuple[object, ...] | None,
    ) -> tuple[object, ...]:
        key = _add_suite_node(
            forest,
            domain=domain,
            kind=kind,
            path=path,
            qual=qual,
            span=span,
            parent=parent,
        )
        meta = _suite_meta(
            key=key,
            domain=domain,
            kind=kind,
            path=path,
            qual=qual,
            span=span,
            parent=parent,
        )
        suite_meta_index[key] = meta
        row = {"row_kind": "suite", **meta}
        rows.append(row)
        return key

    heading_re = re.compile(r"^(#{1,6})\s+(.*)$")
    list_item_re = re.compile(r"^\s*[-*+]\s+")
    issue_checklist_re = re.compile(r"^\s*[-*+]\s*\[(?P<state>[ xX])\]\s+(?P<text>.*)$")

    for path in doc_paths:
        check_deadline()
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        doc_span = (0, 0, max(len(lines) - 1, 0), len(lines[-1]) if lines else 0)
        fm_block = _frontmatter_block(lines)
        fm_payload: Frontmatter = {}
        fm_end = None
        if fm_block is not None:
            fm_lines, fm_end = fm_block
            fm_payload, _yaml_error = _parse_yaml_frontmatter(list(fm_lines))
        if not fm_payload:
            missing_frontmatter.add(rel)
        else:
            docs[rel] = Doc(frontmatter=fm_payload, body="\n".join(lines[fm_end + 1 :] if fm_end is not None else lines))
        doc_id = fm_payload.get("doc_id") if isinstance(fm_payload.get("doc_id"), str) else None
        doc_qual = doc_id or rel

        doc_suite = _record_suite(
            domain="docflow",
            kind="doc_file",
            path=rel,
            qual=doc_qual,
            span=doc_span,
            parent=None,
        )
        doc_suite_by_rel[rel] = doc_suite

        if fm_end is not None:
            front_span = (0, 0, fm_end, len(lines[fm_end]) if fm_end < len(lines) else 0)
            front_suite = _record_suite(
                domain="docflow",
                kind="frontmatter",
                path=rel,
                qual=doc_qual,
                span=front_span,
                parent=doc_suite,
            )
            base = suite_meta_index[front_suite]
            for field, value in fm_payload.items():
                check_deadline()
                if isinstance(value, dict):
                    for entry_key, entry_value in value.items():
                        check_deadline()
                        rows.append(
                            {
                                "row_kind": "frontmatter_entry",
                                **base,
                                "field": field,
                                "entry_key": entry_key,
                                "entry_value": entry_value,
                            }
                        )
                elif isinstance(value, list):
                    for item in value:
                        check_deadline()
                        rows.append(
                            {
                                "row_kind": "frontmatter_item",
                                **base,
                                "field": field,
                                "value": item,
                            }
                        )
                else:
                    rows.append(
                        {
                            "row_kind": "frontmatter_field",
                            **base,
                            "field": field,
                            "value": value,
                        }
                    )

        headings: list[tuple[int, int, str]] = []
        for idx, line in enumerate(lines):
            check_deadline()
            match = heading_re.match(line.strip())
            if match:
                headings.append((idx, len(match.group(1)), match.group(2).strip()))
        section_ranges: list[tuple[int, int, tuple[object, ...]]] = []
        for index, (start_line, level, title) in enumerate(headings):
            check_deadline()
            end_line = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines) - 1
            section_span = (start_line, 0, max(end_line, start_line), len(lines[end_line]) if lines else 0)
            section_qual = f"{doc_qual}#{_slugify_heading(title)}"
            section_suite = _record_suite(
                domain="docflow",
                kind="section",
                path=rel,
                qual=section_qual,
                span=section_span,
                parent=doc_suite,
            )
            rows.append(
                {
                    "row_kind": "section_meta",
                    **suite_meta_index[section_suite],
                    "heading": title,
                    "level": level,
                }
            )
            section_ranges.append((start_line, end_line, section_suite))

        def _section_for_line(line_no: int) -> tuple[object, ...] | None:
            for start, end, suite_key in reversed(section_ranges):
                check_deadline()
                if start <= line_no <= end:
                    return suite_key
            return None

        sppf_issue_re = re.compile(r"\bGH-(\d+)\b")
        for idx, line in enumerate(lines):
            check_deadline()
            if not list_item_re.match(line):
                continue
            parent_suite = _section_for_line(idx) or doc_suite
            item_span = (idx, 0, idx, len(line))
            item_qual = f"{doc_qual}::item:{idx + 1}"
            item_suite = _record_suite(
                domain="docflow",
                kind="list_item",
                path=rel,
                qual=item_qual,
                span=item_span,
                parent=parent_suite,
            )
            rows.append(
                {
                    "row_kind": "list_item_meta",
                    **suite_meta_index[item_suite],
                    "text": line.strip(),
                }
            )
            if "sppf{" in line:
                tag_match = SPPF_TAG_RE.search(line)
                tags = _parse_sppf_tag(tag_match.group(1)) if tag_match else {}
                sppf_span = (idx, 0, idx, len(line))
                sppf_qual = f"{doc_qual}::sppf:{idx + 1}"
                sppf_suite = _record_suite(
                    domain="docflow",
                    kind="sppf_item",
                    path=rel,
                    qual=sppf_qual,
                    span=sppf_span,
                    parent=item_suite,
                )
                issues_found = [f"GH-{issue_id}" for issue_id in sppf_issue_re.findall(line)]
                rows.append(
                    {
                        "row_kind": "sppf_item",
                        **suite_meta_index[sppf_suite],
                        "checklist_state": SPPF_LINE_RE.match(line.strip()).group("state")
                        if SPPF_LINE_RE.match(line.strip())
                        else "",
                        "doc_status": tags.get("doc"),
                        "impl_status": tags.get("impl"),
                        "doc_ref": tags.get("doc_ref"),
                        "issue_refs": issues_found,
                    }
                )
                for issue_key in issues_found:
                    check_deadline()
                    issue_meta = issues.get(issue_key)
                    if issue_meta is None:
                        continue
                    issue_suite_key = _suite_key(
                        domain="github",
                        kind="issue",
                        path=issue_key,
                        qual=issue_key,
                        span=(0, 0, 0, 0),
                    )
                    issue_node = forest.add_node(
                        "SuiteSite",
                        issue_suite_key,
                        meta={
                            "domain": "github",
                            "kind": "issue",
                            "path": issue_key,
                            "qual": issue_key,
                            "span": [0, 0, 0, 0],
                        },
                    )
                    forest.add_alt("SuiteRef", (forest.add_node("SuiteSite", sppf_suite, meta=None), issue_node))

        for idx, line in enumerate(lines):
            check_deadline()
            if "sppf{" not in line:
                continue
            if list_item_re.match(line):
                continue
            parent_suite = _section_for_line(idx) or doc_suite
            tag_match = SPPF_TAG_RE.search(line)
            tags = _parse_sppf_tag(tag_match.group(1)) if tag_match else {}
            sppf_span = (idx, 0, idx, len(line))
            sppf_qual = f"{doc_qual}::sppf:{idx + 1}"
            sppf_suite = _record_suite(
                domain="docflow",
                kind="sppf_item",
                path=rel,
                qual=sppf_qual,
                span=sppf_span,
                parent=parent_suite,
            )
            issues_found = [f"GH-{issue_id}" for issue_id in sppf_issue_re.findall(line)]
            rows.append(
                {
                    "row_kind": "sppf_item",
                    **suite_meta_index[sppf_suite],
                    "checklist_state": SPPF_LINE_RE.match(line.strip()).group("state")
                    if SPPF_LINE_RE.match(line.strip())
                    else "",
                    "doc_status": tags.get("doc"),
                    "impl_status": tags.get("impl"),
                    "doc_ref": tags.get("doc_ref"),
                    "issue_refs": issues_found,
                }
            )

    revisions: dict[str, int] = {}
    for rel, payload in docs.items():
        check_deadline()
        doc_rev = payload.frontmatter.get("doc_revision")
        if isinstance(doc_rev, int):
            revisions[rel] = doc_rev
        _add_section_revisions(revisions, rel=rel, fm=payload.frontmatter)

    def _suite_base_meta(rel: str, doc_id: str | None) -> dict[str, object]:
        suite_key = doc_suite_by_rel.get(rel)
        if suite_key is not None and suite_key in suite_meta_index:
            return suite_meta_index[suite_key]
        return _docflow_base_meta(rel, doc_id)

    invariant_rows, _ = _docflow_invariant_rows(
        docs=docs,
        revisions=revisions,
        core_set=set(CORE_GOVERNANCE_DOCS),
        missing_frontmatter=missing_frontmatter,
        base_meta=_suite_base_meta,
    )
    rows.extend(invariant_rows)

    for issue_key, meta in issues.items():
        check_deadline()
        issue_suite = _record_suite(
            domain="github",
            kind="issue",
            path=issue_key,
            qual=issue_key,
            span=(0, 0, 0, 0),
            parent=None,
        )
        base = suite_meta_index[issue_suite]
        rows.append(
            {
                "row_kind": "issue_meta",
                **base,
                "issue_id": issue_key,
                "title": meta.get("title"),
                "state": meta.get("state"),
                "url": meta.get("url"),
            }
        )
        body = meta.get("body")
        if isinstance(body, str) and body.strip():
            body_lines = body.splitlines()
            body_span = (
                0,
                0,
                max(len(body_lines) - 1, 0),
                len(body_lines[-1]) if body_lines else 0,
            )
            body_suite = _record_suite(
                domain="github",
                kind="issue_body",
                path=issue_key,
                qual=f"{issue_key}::body",
                span=body_span,
                parent=issue_suite,
            )
            rows.append(
                {
                    "row_kind": "issue_body",
                    **suite_meta_index[body_suite],
                    "line_count": len(body_lines),
                }
            )
            for idx, line in enumerate(body_lines):
                check_deadline()
                match = issue_checklist_re.match(line)
                if not match:
                    continue
                item_span = (idx, 0, idx, len(line))
                item_suite = _record_suite(
                    domain="github",
                    kind="issue_checklist_item",
                    path=issue_key,
                    qual=f"{issue_key}::item:{idx + 1}",
                    span=item_span,
                    parent=body_suite,
                )
                rows.append(
                    {
                        "row_kind": "issue_checklist_item",
                        **suite_meta_index[item_suite],
                        "checked": match.group("state").lower() == "x",
                        "text": match.group("text").strip(),
                    }
                )
        labels = meta.get("labels")
        if isinstance(labels, list):
            for label in labels:
                check_deadline()
                if isinstance(label, dict) and isinstance(label.get("name"), str):
                    rows.append(
                        {
                            "row_kind": "issue_label",
                            **base,
                            "label": label.get("name"),
                        }
                    )

    rows.sort(
        key=lambda row: (
            str(row.get("row_kind", "")),
            str(row.get("path", "")),
            str(row.get("suite_kind", "")),
            int(row.get("span_line", 0) or 0),
            int(row.get("span_col", 0) or 0),
            str(row.get("qual", "")),
        )
    )

    if forest_output is not None:
        forest_output.parent.mkdir(parents=True, exist_ok=True)
        forest_output.write_text(
            json.dumps(forest.to_wire_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if relation_output is not None:
        relation_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suites": rows,
        }
        relation_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _build_sppf_dependency_graph(root: Path, issues_json: Path | None = None) -> dict[str, object]:
    checklist_path = root / "docs" / "sppf_checklist.md"
    if not checklist_path.exists():
        raise FileNotFoundError(checklist_path)
    text = checklist_path.read_text(encoding="utf-8")
    _, body = _parse_frontmatter(text)
    issue_re = re.compile(r"\bGH-(\d+)\b")
    issues_meta = _load_issues_json(issues_json)
    issue_nodes: dict[str, dict[str, object]] = {}
    doc_nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    issues_without_doc_ref: set[str] = set()
    docs_without_issue: set[str] = set()

    for lineno, raw in enumerate(body.splitlines(), start=1):
        check_deadline()
        line = raw.strip()
        match = SPPF_LINE_RE.match(line)
        if not match:
            continue
        issue_ids = issue_re.findall(line)
        if not issue_ids:
            continue
        tag_match = SPPF_TAG_RE.search(line)
        tags = _parse_sppf_tag(tag_match.group(1)) if tag_match else {}
        doc_refs: list[tuple[str, int | None]] = []
        doc_ref_raw = tags.get("doc_ref")
        if doc_ref_raw:
            doc_refs = _parse_doc_ref(doc_ref_raw)
        formatted_refs = [_format_doc_ref(doc_id, rev) for doc_id, rev in doc_refs]
        for issue_id in issue_ids:
            check_deadline()
            issue_key = f"GH-{issue_id}"
            node = issue_nodes.setdefault(issue_key, {
                "id": issue_key,
                "checklist_state": match.group("state"),
                "doc_status": tags.get("doc"),
                "impl_status": tags.get("impl"),
                "line": raw,
                "line_no": lineno,
                "doc_refs": [],
            })
            # merge metadata if provided
            meta = issues_meta.get(issue_key)
            if meta:
                node.setdefault("title", meta.get("title"))
                node.setdefault("state", meta.get("state"))
                labels = meta.get("labels")
                if isinstance(labels, list):
                    node.setdefault("labels", [lab.get("name") for lab in labels if isinstance(lab, dict)])
            if formatted_refs:
                node["doc_refs"] = _sorted(set(node.get("doc_refs", [])) | set(formatted_refs))
            else:
                issues_without_doc_ref.add(issue_key)

            for doc_id, rev in doc_refs:
                check_deadline()
                doc_key = _format_doc_ref(doc_id, rev)
                doc_node = doc_nodes.setdefault(doc_key, {
                    "id": doc_key,
                    "doc_id": doc_id,
                    "revision": rev,
                    "issues": [],
                })
                doc_node["issues"] = _sorted(set(doc_node.get("issues", [])) | {issue_key})
                edges.append({"from": doc_key, "to": issue_key, "kind": "doc_ref"})

        if formatted_refs and not issue_ids:
            for doc_id, rev in doc_refs:
                check_deadline()
                docs_without_issue.add(_format_doc_ref(doc_id, rev))

    graph = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "docs/sppf_checklist.md",
        "issues": issue_nodes,
        "docs": doc_nodes,
        "edges": edges,
        "issues_without_doc_ref": _sorted(issues_without_doc_ref),
        "docs_without_issue": _sorted(docs_without_issue),
    }
    return graph


def _write_sppf_graph_outputs(
    graph: dict[str, JSONValue],
    *,
    json_output: Path | None,
    dot_output: Path | None,
) -> None:
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    if dot_output is not None:
        dot_output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["digraph sppf_deps {"]
        for doc_key, node in _sorted(graph.get("docs", {}).items()):
            check_deadline()
            label = doc_key
            lines.append(f"  \"{doc_key}\" [shape=box,label=\"{label}\"];")
        for issue_key, node in _sorted(graph.get("issues", {}).items()):
            check_deadline()
            label = issue_key
            title = node.get("title")
            if isinstance(title, str):
                label = f"{issue_key}\\n{title}"
            lines.append(f"  \"{issue_key}\" [shape=ellipse,label=\"{label}\"];")
        for edge in graph.get("edges", []):
            check_deadline()
            src = edge.get("from")
            dst = edge.get("to")
            if src and dst:
                lines.append(f"  \"{src}\" -> \"{dst}\";")
        lines.append("}")
        dot_output.write_text("\n".join(lines), encoding="utf-8")


def _influence_statuses(root: Path) -> dict[str, str]:
    index_path = GOVERNANCE_PATHS.influence_index_path(root=root)
    if not index_path.exists():
        return {}
    text = index_path.read_text(encoding="utf-8")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        check_deadline()
        match = re.match(r"^- in/(in-\d+)\.md\s+—\s+", line.strip())
        if not match:
            continue
        status_match = re.search(r"\*\*(\w+)\*\*", line)
        if not status_match:
            continue
        doc_id = match.group(1)
        entries[doc_id] = status_match.group(1).lower()
    return entries


def _extract_in_status_declaration(text: str) -> tuple[str | None, int | None, str | None, bool]:
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        check_deadline()
        line = raw.strip()
        if line.lower().startswith("status:"):
            payload = line.split(":", 1)[1].strip()
            norm = _normalize_in_status_declaration(payload)
            return norm, idx + 1, payload, STATUS_TRIPLET_OVERRIDE_MARKER in raw
    for idx, raw in enumerate(lines):
        check_deadline()
        if raw.strip().lower() != "### status":
            continue
        cursor = idx + 1
        while cursor < len(lines) and not lines[cursor].strip():
            check_deadline()
            cursor += 1
        if cursor >= len(lines):
            return None, idx + 1, None, STATUS_TRIPLET_OVERRIDE_MARKER in raw
        payload = lines[cursor].strip()
        norm = _normalize_in_status_declaration(payload)
        window = "\n".join(lines[idx : min(cursor + 2, len(lines))])
        return norm, cursor + 1, payload, STATUS_TRIPLET_OVERRIDE_MARKER in window
    return None, None, None, False


def _normalize_in_status_declaration(value: str) -> str | None:
    lowered = value.strip().lower()
    if not lowered:
        return None
    if "adopted" in lowered or "done" in lowered:
        return "adopted"
    if "partial" in lowered:
        return "partial"
    if "queued" in lowered:
        return "queued"
    if "rejected" in lowered or "out of scope" in lowered:
        return "rejected"
    if "planned" in lowered or "draft" in lowered:
        return "planned"
    return None


def _collect_checklist_statuses(root: Path) -> dict[str, dict[str, object]]:
    checklist_path = root / "docs" / "sppf_checklist.md"
    if not checklist_path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for lineno, raw in enumerate(checklist_path.read_text(encoding="utf-8").splitlines(), start=1):
        check_deadline()
        line = raw.strip()
        match = SPPF_LINE_RE.match(line)
        if match:
            state_token = match.group("state")
        else:
            loose = re.search(r"\[(x|~| )\]", line)
            if not loose:
                continue
            state_token = loose.group(1)
        state = CHECKLIST_STATE_MAP.get(state_token)
        if state is None:
            continue
        doc_ids: set[str] = set()
        tag_match = SPPF_TAG_RE.search(line)
        if tag_match:
            tags = _parse_sppf_tag(tag_match.group(1))
            ref_value = tags.get("doc_ref")
            if ref_value:
                for doc_id, _ in _parse_doc_ref(ref_value):
                    check_deadline()
                    if doc_id.startswith("in-") and doc_id[3:].isdigit():
                        doc_ids.add(doc_id)
        for found in re.findall(r"\bin-(\d+)\b", line):
            check_deadline()
            doc_ids.add(f"in-{int(found)}")
        for doc_id in doc_ids:
            check_deadline()
            existing = records.get(doc_id)
            if existing is None:
                records[doc_id] = {
                    "status": state,
                    "line_no": lineno,
                    "line": raw.strip(),
                    "override": STATUS_TRIPLET_OVERRIDE_MARKER in raw,
                }
                continue
            if existing.get("status") == state:
                if bool(existing.get("override")) or STATUS_TRIPLET_OVERRIDE_MARKER in raw:
                    existing["override"] = True
                continue
            existing["status"] = "mixed"
            existing["line"] = f"{existing.get('line')} || {raw.strip()}"
            existing["override"] = bool(existing.get("override")) or STATUS_TRIPLET_OVERRIDE_MARKER in raw
    return records


def _collect_influence_rows(root: Path) -> dict[str, dict[str, object]]:
    index_path = root / "docs" / "influence_index.md"
    if not index_path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    for lineno, raw in enumerate(index_path.read_text(encoding="utf-8").splitlines(), start=1):
        check_deadline()
        line = raw.strip()
        match = re.match(r"^- in/(in-\d+)\.md\s+—\s+", line)
        if not match:
            continue
        status_match = re.search(r"\*\*(\w+)\*\*", raw)
        if not status_match:
            continue
        rows[match.group(1)] = {
            "status": status_match.group(1).lower(),
            "line_no": lineno,
            "line": raw.strip(),
            "override": STATUS_TRIPLET_OVERRIDE_MARKER in raw,
        }
    return rows


def _sppf_status_triplet_violations(root: Path) -> list[str]:
    in_records: dict[str, dict[str, object]] = {}
    for path in _sorted((root / "in").glob("in-*.md")):
        check_deadline()
        norm, line_no, raw, override = _extract_in_status_declaration(path.read_text(encoding="utf-8"))
        in_records[path.stem] = {
            "status": norm,
            "line_no": line_no,
            "line": raw,
            "override": override,
        }

    checklist_records = _collect_checklist_statuses(root)
    influence_records = _collect_influence_rows(root)
    violations: list[str] = []

    doc_ids = _sorted(in_records)
    for doc_id in doc_ids:
        check_deadline()
        in_rec = in_records.get(doc_id)
        checklist_rec = checklist_records.get(doc_id)
        influence_rec = influence_records.get(doc_id)
        if in_rec is None or checklist_rec is None or influence_rec is None:
            continue
        if not isinstance(in_rec.get("status"), str) or not isinstance(checklist_rec.get("status"), str):
            continue
        if not isinstance(influence_rec.get("status"), str):
            continue
        declared = str(in_rec["status"])
        checklist_status = str(checklist_rec["status"])
        influence_status = str(influence_rec["status"])
        has_override = bool(in_rec.get("override")) or bool(checklist_rec.get("override")) or bool(influence_rec.get("override"))
        if has_override:
            continue

        allowed_influence = DECLARATION_TO_INFLUENCE_MAP.get(declared, set())
        allowed_checklist = DECLARATION_TO_CHECKLIST_MAP.get(declared, set())
        expected_checklist = INFLUENCE_STATUS_MAP.get(influence_status)
        consistent = (
            influence_status in allowed_influence
            and checklist_status in allowed_checklist
            and (expected_checklist is None or checklist_status == expected_checklist)
        )
        if consistent:
            continue

        in_line = in_rec.get("line_no")
        check_line = checklist_rec.get("line_no")
        index_line = influence_rec.get("line_no")
        in_ref = f"in/{doc_id}.md:{in_line}" if in_line else f"in/{doc_id}.md"
        check_ref = f"docs/sppf_checklist.md:{check_line}" if check_line else "docs/sppf_checklist.md"
        index_ref = f"docs/influence_index.md:{index_line}" if index_line else "docs/influence_index.md"
        violations.append(
            "status-triplet conflict for "
            f"{doc_id}: {in_ref} declares={declared!r}; "
            f"{check_ref} checklist={checklist_status!r}; "
            f"{index_ref} summary={influence_status!r}. "
            f"Use marker '{STATUS_TRIPLET_OVERRIDE_MARKER}' on one of these records to acknowledge an intentional divergence."
        )
    return violations


def _in_doc_revisions(root: Path) -> dict[str, int]:
    revisions: dict[str, int] = {}
    inbox = GOVERNANCE_PATHS.in_dir(root=root)
    if not inbox.exists():
        return revisions
    for path in inbox.glob("in-*.md"):
        check_deadline()
        text = path.read_text(encoding="utf-8")
        fm, _ = _parse_frontmatter(text)
        doc_rev = fm.get("doc_revision")
        if isinstance(doc_rev, int):
            revisions[path.stem] = doc_rev
    return revisions


def _doc_revision_for_ref(root: Path, doc_id: str) -> int | None:
    if doc_id.startswith("in-") and doc_id[3:].isdigit():
        path = GOVERNANCE_PATHS.in_dir(root=root) / f"{doc_id}.md"
    else:
        path = root / doc_id
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    fm, _ = _parse_frontmatter(text)
    doc_rev = fm.get("doc_revision")
    return doc_rev if isinstance(doc_rev, int) else None


def _sppf_axis_audit(root: Path, docs: dict[str, Doc]) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []
    sppf_doc = docs.get("docs/sppf_checklist.md")
    if sppf_doc is None:
        return violations, warnings
    statuses = _influence_statuses(root)
    revisions = _in_doc_revisions(root)
    for raw in sppf_doc.body.splitlines():
        check_deadline()
        line = raw.strip()
        match = SPPF_LINE_RE.match(line)
        if not match:
            continue
        has_in_ref = bool(SPPF_IN_REF_RE.search(line))
        tag_match = SPPF_TAG_RE.search(line)
        if has_in_ref and not tag_match:
            violations.append(f"docs/sppf_checklist.md: missing sppf{{...}} tag: {line}")
            continue
        if not tag_match:
            continue
        tags = _parse_sppf_tag(tag_match.group(1))
        doc_status = tags.get("doc")
        impl_status = tags.get("impl")
        doc_ref = tags.get("doc_ref")
        if not doc_status or doc_status not in SPPF_ALLOWED_STATUSES:
            violations.append(f"docs/sppf_checklist.md: invalid doc status in sppf tag: {line}")
        if not impl_status or impl_status not in SPPF_ALLOWED_STATUSES:
            violations.append(f"docs/sppf_checklist.md: invalid impl status in sppf tag: {line}")
        if not doc_ref:
            violations.append(f"docs/sppf_checklist.md: missing doc_ref in sppf tag: {line}")
            continue
        refs = _parse_doc_ref(doc_ref)
        if not refs:
            violations.append(f"docs/sppf_checklist.md: invalid doc_ref in sppf tag: {line}")
            continue
        expected_doc_statuses: set[str] = set()
        checked_doc_status = False
        for doc_id, rev in refs:
            check_deadline()
            if doc_id.startswith("in-") and doc_id[3:].isdigit():
                expected = statuses.get(doc_id)
                if expected is None:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id} missing in docs/influence_index.md: {line}"
                    )
                    continue
                mapped = INFLUENCE_STATUS_MAP.get(expected)
                if mapped is None:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id} has unknown status {expected}: {line}"
                    )
                    continue
                expected_doc_statuses.add(mapped)
                checked_doc_status = True
                actual_rev = revisions.get(doc_id)
                if actual_rev is None:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id} missing in/ doc revision: {line}"
                    )
                elif rev is None or rev != actual_rev:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id}@{rev} does not match in/{doc_id}.md rev {actual_rev}: {line}"
                    )
            else:
                actual_rev = _doc_revision_for_ref(root, doc_id)
                if actual_rev is None:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id} missing doc revision: {line}"
                    )
                elif rev is None or rev != actual_rev:
                    violations.append(
                        f"docs/sppf_checklist.md: doc_ref {doc_id}@{rev} does not match {doc_id} rev {actual_rev}: {line}"
                    )
        if checked_doc_status and doc_status and expected_doc_statuses and doc_status not in expected_doc_statuses:
            violations.append(
                f"docs/sppf_checklist.md: doc status {doc_status} does not match influence index {_sorted(expected_doc_statuses)}: {line}"
            )
        state = match.group("state")
        if state == "x" and (doc_status != "done" or impl_status != "done"):
            violations.append(
                f"docs/sppf_checklist.md: [x] requires doc=done and impl=done: {line}"
            )
        if state != "x" and doc_status == "done" and impl_status == "done":
            warnings.append(
                f"docs/sppf_checklist.md: doc+impl done but not marked [x]: {line}"
            )
    return violations, warnings


def _frontmatter_block_from_text(text: str) -> tuple[list[str], str] | None:
    carrier = parse_frontmatter_document(text)
    if not carrier.has_closed_block:
        return None
    return list(carrier.raw_frontmatter_lines), carrier.body


def _parse_yaml_frontmatter(lines: list[str]) -> tuple[Frontmatter, str | None]:
    text = "---\n" + "\n".join(lines) + "\n---\n"
    carrier = parse_frontmatter_document(text)
    if carrier.mode is FrontmatterParseMode.YAML:
        normalized: Frontmatter = {}
        for key, value in carrier.payload.items():
            check_deadline()
            normalized[key] = cast(FrontmatterValue, value)
        return normalized, None
    return {}, carrier.detail or "invalid YAML frontmatter"


def _parse_frontmatter_with_mode(text: str) -> tuple[Frontmatter, str, str, str | None]:
    carrier = parse_frontmatter_document(text)
    normalized: Frontmatter = {}
    for key, value in carrier.payload.items():
        normalized[key] = cast(FrontmatterValue, value)
    return normalized, carrier.body, carrier.mode.value, carrier.detail


def _frontmatter_parse_warning(
    *,
    path: str,
    mode: str,
    detail: str | None,
) -> str | None:
    if mode == "yaml_parse_failed":
        suffix = f" ({detail})" if detail else ""
        return (
            f"{path}: frontmatter failed strict YAML parsing{suffix}; "
            "document may be skipped from governance invariants"
        )
    return None


def _parse_frontmatter(text: str) -> tuple[Frontmatter, str]:
    payload, body, _mode, _detail = _parse_frontmatter_with_mode(text)
    return payload, body


def _docflow_base_meta(rel: str, doc_id: str | None) -> dict[str, object]:
    return {"path": rel, "qual": doc_id or rel}


def _split_doc_ref(ref: str) -> tuple[str, str | None]:
    if "#" in ref:
        base, frag = ref.split("#", 1)
        frag = frag.strip()
        return base, frag or None
    return ref, None


def _doc_ref_base(ref: str) -> str:
    return _split_doc_ref(ref)[0]


def _review_note_mentions_expected_revisions(
    *,
    note: str,
    expected_doc_revision: int,
    expected_section_revision: int | None,
) -> bool:
    if f"rev{expected_doc_revision}" not in note:
        return False
    if expected_section_revision is None:
        return True
    return f"section v{expected_section_revision}" in note


def _add_section_revisions(
    revisions: dict[str, int],
    *,
    rel: str,
    fm: Frontmatter,
) -> None:
    sections = fm.get("doc_sections")
    if not isinstance(sections, dict):
        return
    for key, value in sections.items():
        check_deadline()
        if not isinstance(key, str) or not key:
            continue
        if not isinstance(value, int):
            continue
        revisions[f"{rel}#{key}"] = value


def _docflow_invariant_rows(
    *,
    docs: dict[str, Doc],
    revisions: dict[str, int],
    core_set: set[str],
    missing_frontmatter: set[str],
    implication_docs: dict[str, Doc] | None = None,
    base_meta: Callable[[str, str | None], dict[str, object]] = _docflow_base_meta,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    control_loop_doc = docs.get(GOVERNANCE_CONTROL_LOOPS_DOC)
    declared_domains: set[str] = set()
    if control_loop_doc is not None:
        fm_domains = control_loop_doc.frontmatter.get("loop_domains")
        if isinstance(fm_domains, list):
            declared_domains = {
                str(item).strip()
                for item in fm_domains
                if isinstance(item, str) and item.strip()
            }
    for rel in _sorted(missing_frontmatter):
        check_deadline()
        rows.append(
            {
                "row_kind": "doc_missing_frontmatter",
                **base_meta(rel, None),
            }
        )
    for rel, payload in docs.items():
        check_deadline()
        fm = payload.frontmatter
        body = payload.body
        doc_id = fm.get("doc_id") if isinstance(fm.get("doc_id"), str) else None
        base = base_meta(rel, doc_id)
        for field in REQUIRED_FIELDS:
            check_deadline()
            rows.append(
                {
                    "row_kind": "doc_required_field",
                    **base,
                    "field": field,
                    "present": field in fm,
                }
            )
        for field in ("doc_scope", "doc_requires"):
            check_deadline()
            if field in fm:
                rows.append(
                    {
                        "row_kind": "doc_field_type",
                        **base,
                        "field": field,
                        "expected": "list",
                        "valid": isinstance(fm.get(field), list),
                    }
                )
        for field in (
            "doc_reviewed_as_of",
            "doc_review_notes",
            "doc_sections",
            "doc_section_requires",
            "doc_section_reviews",
        ):
            check_deadline()
            if field in fm:
                rows.append(
                    {
                        "row_kind": "doc_field_type",
                        **base,
                        "field": field,
                        "expected": "map",
                        "valid": isinstance(fm.get(field), dict),
                    }
                )
        requires = fm.get("doc_requires", [])
        requires_list = requires if isinstance(requires, list) else []
        if isinstance(requires, list):
            for req in requires:
                check_deadline()
                if not isinstance(req, str):
                    continue
                explicit = req in body
                implicit = False
                if not explicit:
                    req_name = _lower_name(Path(req))
                    if req_name and req_name in body.lower():
                        implicit = True
                        warnings.append(
                            f"{rel}: implicit reference to {req} (Tier-2); prefer explicit path"
                        )
                rows.append(
                    {
                        "row_kind": "doc_requires_ref",
                        **base,
                        "req": req,
                        "explicit": explicit,
                        "implicit": implicit,
                    }
                )
        if fm.get("doc_authority") == "normative":
            requires_raw = fm.get("doc_requires", [])
            requires = set(_doc_ref_base(req) for req in requires_raw if isinstance(req, str))
            required_core = core_set
            projection = fm.get("doc_dependency_projection")
            if isinstance(projection, str) and projection == "glossary_root":
                required_core = set()
            missing = _sorted(required_core - {rel} - requires)
            for req in missing:
                check_deadline()
                rows.append(
                    {
                        "row_kind": "doc_missing_governance_ref",
                        **base,
                        "missing": req,
                    }
                )
        reviewed = fm.get("doc_reviewed_as_of")
        review_notes = fm.get("doc_review_notes")
        if rel in _REVIEW_NOTE_REVISION_LINT_DOCS and requires_list:
            for req in requires_list:
                check_deadline()
                if not isinstance(req, str):
                    continue
                note = review_notes.get(req) if isinstance(review_notes, dict) else None
                if not isinstance(note, str) or not note.strip():
                    continue
                req_base, req_anchor = _split_doc_ref(req)
                dep_doc = docs.get(req_base)
                if dep_doc is None:
                    continue
                dep_doc_revision = dep_doc.frontmatter.get("doc_revision")
                if not isinstance(dep_doc_revision, int):
                    continue
                expected_section_revision = revisions.get(req) if req_anchor is not None else None
                if req_anchor is not None and not isinstance(expected_section_revision, int):
                    continue
                rows.append(
                    {
                        "row_kind": "doc_review_note_revision",
                        **base,
                        "req": req,
                        "expected_doc_revision": dep_doc_revision,
                        "expected_section_revision": expected_section_revision,
                        "match": _review_note_mentions_expected_revisions(
                            note=note,
                            expected_doc_revision=dep_doc_revision,
                            expected_section_revision=expected_section_revision,
                        ),
                    }
                )
        if isinstance(requires, list) and requires:
            for req in requires:
                check_deadline()
                if not isinstance(req, str):
                    continue
                expected = revisions.get(req)
                seen = reviewed.get(req) if isinstance(reviewed, dict) else None
                resolved = expected is not None
                match = isinstance(seen, int) and expected is not None and seen == expected
                rows.append(
                    {
                        "row_kind": "doc_review_pin",
                        **base,
                        "req": req,
                        "expected": expected,
                        "seen": seen,
                        "resolved": resolved,
                        "match": match,
                    }
                )
                note = review_notes.get(req) if isinstance(review_notes, dict) else None
                rows.append(
                    {
                        "row_kind": "doc_review_note",
                        **base,
                        "req": req,
                        "note_present": isinstance(note, str) and bool(note.strip()),
                    }
                )
        commutes = fm.get("doc_commutes_with")
        if isinstance(commutes, list):
            for other in commutes:
                check_deadline()
                if not isinstance(other, str):
                    continue
                other_base = _doc_ref_base(other)
                other_doc = docs.get(other_base)
                target_exists = other_doc is not None
                reciprocated = False
                if target_exists:
                    other_commutes = other_doc.frontmatter.get("doc_commutes_with", [])
                    if isinstance(other_commutes, list):
                        reciprocated = any(
                            _doc_ref_base(item) == rel
                            for item in other_commutes
                            if isinstance(item, str)
                        )
                rows.append(
                    {
                        "row_kind": "doc_commute_edge",
                        **base,
                        "other": other,
                        "target_exists": target_exists,
                        "reciprocated": reciprocated,
                    }
                )
    loop_base = base_meta(
        GOVERNANCE_CONTROL_LOOPS_DOC,
        (
            control_loop_doc.frontmatter.get("doc_id")
            if control_loop_doc is not None
            and isinstance(control_loop_doc.frontmatter.get("doc_id"), str)
            else None
        ),
    )
    for domain in NORMATIVE_LOOP_DOMAINS:
        check_deadline()
        rows.append(
            {
                "row_kind": "doc_loop_entry",
                **loop_base,
                "domain": domain,
                "required": True,
                "declared": domain in declared_domains,
            }
        )

    matrix_doc_rel = "docs/governance_loop_matrix.md"
    matrix_doc = docs.get(matrix_doc_rel)
    matrix_gate_ids = _matrix_gate_ids_from_markdown(matrix_doc.body) if matrix_doc is not None else set()
    matrix_doc_id = (
        matrix_doc.frontmatter.get("doc_id")
        if matrix_doc is not None and isinstance(matrix_doc.frontmatter.get("doc_id"), str)
        else None
    )
    matrix_base = base_meta(matrix_doc_rel, matrix_doc_id)
    for gate_id in _sorted(load_governance_rules().gates.keys()):
        check_deadline()
        rows.append(
            {
                "row_kind": "doc_loop_matrix_gate",
                **matrix_base,
                "gate_id": gate_id,
                "required": True,
                "declared": gate_id in matrix_gate_ids,
            }
        )

    matrix_source_docs = implication_docs if implication_docs is not None else docs
    per_doc_lattices: dict[str, dict[str, JSONValue]] = {}
    for rel, payload in matrix_source_docs.items():
        check_deadline()
        notions = _extract_doc_body_notions_with_anchors(path=rel, doc=payload)
        per_doc_lattices[rel] = _build_doc_implication_lattice(path=rel, notions=notions)
    _composed_lattices, matrix_warnings, matrix_conflicts = _compose_doc_dependency_matrices(
        docs=matrix_source_docs,
        per_doc_lattices=per_doc_lattices,
    )
    rows.extend(matrix_conflicts)
    warnings.extend(matrix_warnings)

    return rows, warnings


def _load_test_evidence(root: Path) -> dict[str, object] | None:
    candidates = [
        root / "out" / "test_evidence.json",
        root / "artifacts" / "out" / "test_evidence.json",
    ]
    for path in candidates:
        check_deadline()
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _evidence_rows_from_test_evidence(payload: dict[str, JSONValue]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    evidence_index = payload.get("evidence_index")
    if not isinstance(evidence_index, list):
        return rows
    for entry in evidence_index:
        check_deadline()
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not isinstance(key, dict):
            continue
        normalized = evidence_keys.normalize_key(key)
        evidence_id = evidence_keys.key_identity(normalized)
        display = entry.get("display")
        if not isinstance(display, str) or not display.strip():
            display = evidence_keys.render_display(normalized)
        rows.append(
            {
                "row_kind": "evidence_key",
                "evidence_id": evidence_id,
                "evidence_kind": normalized.get("k"),
                "evidence_key": normalized,
                "evidence_display": display,
                "evidence_source": "test_evidence",
                "tests": entry.get("tests", []),
            }
        )
    return rows


def _impact_rows(root: Path) -> list[dict[str, object]]:
    index = build_impact_index(root=root)
    rows: list[dict[str, object]] = []
    for link in index.links:
        check_deadline()
        rows.append(
            {
                "row_kind": "impact_link",
                "path": link.source,
                "qual": link.source,
                "impact_source_kind": link.source_kind,
                "impact_target": link.target,
                "impact_confidence": link.confidence,
            }
        )
    return rows


def _format_doc_missing_frontmatter_violation(_row: Mapping[str, JSONValue], path: str) -> str:
    return f"{path}: missing frontmatter"


def _format_doc_required_field_violation(row: Mapping[str, JSONValue], path: str) -> str:
    field = row.get("field", "?")
    return f"{path}: missing frontmatter field '{field}'"


def _format_doc_field_type_violation(row: Mapping[str, JSONValue], path: str) -> str:
    field = row.get("field", "?")
    expected = row.get("expected", "?")
    return f"{path}: frontmatter field '{field}' must be a {expected}"


def _format_doc_missing_governance_ref_violation(row: Mapping[str, JSONValue], path: str) -> str:
    missing = row.get("missing", "?")
    return f"{path}: missing required governance references: {missing}"


def _format_doc_requires_ref_violation(row: Mapping[str, JSONValue], path: str) -> str:
    req = row.get("req", "?")
    return f"{path}: missing explicit reference to {req}"


def _format_doc_review_pin_violation(row: Mapping[str, JSONValue], path: str) -> str:
    req = row.get("req", "?")
    expected = row.get("expected")
    seen = row.get("seen")
    if not bool(row.get("resolved", False)):
        return f"{path}: doc_reviewed_as_of cannot resolve {req}"
    if not isinstance(seen, int):
        return f"{path}: doc_reviewed_as_of[{req}] must be an integer"
    return f"{path}: doc_reviewed_as_of[{req}]={seen} does not match {expected}"


def _format_doc_review_note_violation(row: Mapping[str, JSONValue], path: str) -> str:
    req = row.get("req", "?")
    return f"{path}: doc_review_notes[{req}] missing or empty"


def _format_doc_review_note_revision_violation(
    row: Mapping[str, JSONValue], path: str
) -> str:
    req = row.get("req", "?")
    expected_doc_revision = row.get("expected_doc_revision", "?")
    expected_section_revision = row.get("expected_section_revision")
    if isinstance(expected_section_revision, int):
        return (
            f"{path}: doc_review_notes[{req}] must mention "
            f"rev{expected_doc_revision} and section v{expected_section_revision}"
        )
    return f"{path}: doc_review_notes[{req}] must mention rev{expected_doc_revision}"


def _format_doc_commute_edge_violation(row: Mapping[str, JSONValue], path: str) -> str:
    other = row.get("other", "?")
    if not bool(row.get("target_exists", False)):
        return f"{path}: doc_commutes_with target missing: {other}"
    return f"{path}: commutation with {other} not reciprocated"


def _format_doc_loop_entry_violation(row: Mapping[str, JSONValue], path: str) -> str:
    domain = row.get("domain", "?")
    return f"{path}: missing governance control-loop declaration for domain: {domain}"


def _format_doc_loop_matrix_gate_violation(row: Mapping[str, JSONValue], path: str) -> str:
    gate_id = row.get("gate_id", "?")
    return f"{path}: governance loop matrix drift; missing gate row for: {gate_id}"


def _format_doc_implication_matrix_conflict_violation(row: Mapping[str, JSONValue], path: str) -> str:
    subject = row.get("subject", "?")
    return f"{path}: implication matrix conflict for notion subject: {subject}"


_DOCFLOW_VIOLATION_FORMATTERS: dict[str, Callable[[Mapping[str, JSONValue], str], str]] = {
    "doc_missing_frontmatter": _format_doc_missing_frontmatter_violation,
    "doc_required_field": _format_doc_required_field_violation,
    "doc_field_type": _format_doc_field_type_violation,
    "doc_missing_governance_ref": _format_doc_missing_governance_ref_violation,
    "doc_requires_ref": _format_doc_requires_ref_violation,
    "doc_review_pin": _format_doc_review_pin_violation,
    "doc_review_note": _format_doc_review_note_violation,
    "doc_review_note_revision": _format_doc_review_note_revision_violation,
    "doc_commute_edge": _format_doc_commute_edge_violation,
    "doc_loop_entry": _format_doc_loop_entry_violation,
    "doc_loop_matrix_gate": _format_doc_loop_matrix_gate_violation,
    "doc_implication_matrix_conflict": _format_doc_implication_matrix_conflict_violation,
}


def _format_docflow_violation(row: Mapping[str, JSONValue]) -> str:
    path = str(row.get("path", "?") or "?")
    kind = str(row.get("row_kind", "") or "")
    formatter = _DOCFLOW_VIOLATION_FORMATTERS.get(kind)
    if formatter is not None:
        return formatter(row, path)
    return f"{path}: docflow invariant violation"


def _match_docflow_rows(
    rows: Iterable[dict[str, object]],
    *,
    matcher: DocflowPredicateMatcher,
    op_registry: Mapping[
        str,
        Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool],
    ],
) -> list[dict[str, object]]:
    matched = list(rows)
    for predicate_name in matcher.predicates:
        check_deadline()
        predicate = op_registry.get(predicate_name)
        if predicate is None:
            never(
                "unknown docflow invariant predicate",
                predicate=predicate_name,
            )
        matched = [
            row
            for row in matched
            if predicate(
                cast(Mapping[str, JSONValue], row),
                matcher.params,
            )
        ]
    return matched


def _docflow_compliance_rows(
    rows: list[dict[str, object]],
    *,
    invariants: Iterable[DocflowInvariant],
) -> list[dict[str, object]]:
    compliance: list[dict[str, object]] = []
    op_registry = _docflow_predicates()
    evidence_rows = [row for row in rows if row.get("row_kind") == "evidence_key"]
    non_evidence_rows = [row for row in rows if row.get("row_kind") != "evidence_key"]
    covered_evidence: set[str] = set()

    def _handle_cover_invariant(
        invariant: DocflowInvariant,
        *,
        matched: list[dict[str, object]],
        evidence_matched: list[dict[str, object]],
        active_flag: bool,
        compliance: list[dict[str, object]],
        covered_evidence: set[str],
    ) -> None:
        del matched
        if not active_flag:
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "invariant": invariant.name,
                    "invariant_kind": invariant.kind,
                    "status": "proposed",
                    "match_count": len(evidence_matched),
                    "detail": "cover target missing" if not evidence_matched else None,
                }
            )
            return
        if evidence_matched:
            for row in evidence_matched:
                check_deadline()
                evidence_id = str(row.get("evidence_id", "") or "")
                if evidence_id:
                    covered_evidence.add(evidence_id)
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "invariant": invariant.name,
                    "invariant_kind": invariant.kind,
                    "status": "compliant",
                    "match_count": len(evidence_matched),
                }
            )
            return
        compliance.append(
            {
                "row_kind": "docflow_compliance",
                "invariant": invariant.name,
                "invariant_kind": invariant.kind,
                "status": "contradicts",
                "match_count": 0,
                "detail": "cover target missing",
            }
        )

    def _handle_never_invariant(
        invariant: DocflowInvariant,
        *,
        matched: list[dict[str, object]],
        evidence_matched: list[dict[str, object]],
        active_flag: bool,
        compliance: list[dict[str, object]],
        covered_evidence: set[str],
    ) -> None:
        del evidence_matched
        del covered_evidence
        if not active_flag:
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "invariant": invariant.name,
                    "invariant_kind": invariant.kind,
                    "status": "proposed",
                    "match_count": len(matched),
                    "would_violate": bool(matched),
                }
            )
            return
        if matched:
            for row in matched:
                check_deadline()
                compliance.append(
                    {
                        "row_kind": "docflow_compliance",
                        "invariant": invariant.name,
                        "invariant_kind": invariant.kind,
                        "status": "contradicts",
                        "match_count": len(matched),
                        "path": row.get("path"),
                        "qual": row.get("qual"),
                        "source_row_kind": row.get("row_kind"),
                    }
                )
            return
        compliance.append(
            {
                "row_kind": "docflow_compliance",
                "invariant": invariant.name,
                "invariant_kind": invariant.kind,
                "status": "compliant",
                "match_count": 0,
            }
        )

    def _handle_require_invariant(
        invariant: DocflowInvariant,
        *,
        matched: list[dict[str, object]],
        evidence_matched: list[dict[str, object]],
        active_flag: bool,
        compliance: list[dict[str, object]],
        covered_evidence: set[str],
    ) -> None:
        del evidence_matched
        del covered_evidence
        if not active_flag:
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "invariant": invariant.name,
                    "invariant_kind": invariant.kind,
                    "status": "proposed",
                    "match_count": len(matched),
                    "would_violate": not bool(matched),
                    "detail": "requirement missing" if not matched else None,
                }
            )
            return
        if matched:
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "invariant": invariant.name,
                    "invariant_kind": invariant.kind,
                    "status": "compliant",
                    "match_count": len(matched),
                }
            )
            return
        compliance.append(
            {
                "row_kind": "docflow_compliance",
                "invariant": invariant.name,
                "invariant_kind": invariant.kind,
                "status": "contradicts",
                "match_count": 0,
                "detail": "requirement missing",
            }
        )

    Handler: TypeAlias = Callable[..., None]
    handlers: dict[str, Handler] = {
        "cover": _handle_cover_invariant,
        "never": _handle_never_invariant,
        "require": _handle_require_invariant,
    }

    for invariant in invariants:
        check_deadline()
        rows_to_match = rows if _invariant_uses_evidence_rows(invariant) else non_evidence_rows
        matched = _match_docflow_rows(
            rows_to_match,
            matcher=invariant.matcher,
            op_registry=op_registry,
        )
        evidence_matched = [
            row for row in matched if row.get("row_kind") == "evidence_key"
        ]
        active_flag = invariant.status == "active"
        handler = handlers.get(invariant.kind)
        if handler is None:
            never("unknown docflow invariant kind", kind=invariant.kind)
        handler(
            invariant,
            matched=matched,
            evidence_matched=evidence_matched,
            active_flag=active_flag,
            compliance=compliance,
            covered_evidence=covered_evidence,
        )
    for row in evidence_rows:
        check_deadline()
        evidence_id = str(row.get("evidence_id", "") or "")
        if evidence_id and evidence_id not in covered_evidence:
            compliance.append(
                {
                    "row_kind": "docflow_compliance",
                    "status": "excess",
                    "evidence_id": evidence_id,
                    "evidence_kind": row.get("evidence_kind"),
                    "evidence_display": row.get("evidence_display"),
                    "evidence_source": row.get("evidence_source"),
                }
            )
    return compliance


def _summarize_docflow_compliance(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {"compliant": 0, "contradicts": 0, "excess": 0, "proposed": 0}
    for row in rows:
        check_deadline()
        status = str(row.get("status", "") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _render_docflow_compliance_md(
    rows: list[dict[str, object]],
    *,
    obligations: DocflowObligationResult | None = None,
) -> list[str]:
    summary = _summarize_docflow_compliance(rows)
    lines: list[str] = []
    lines.append("Docflow compliance report")
    lines.append(f"- compliant: {summary.get('compliant', 0)}")
    lines.append(f"- contradicts: {summary.get('contradicts', 0)}")
    lines.append(f"- proposed: {summary.get('proposed', 0)}")
    lines.append(f"- excess: {summary.get('excess', 0)}")
    if obligations is not None:
        obligation_summary = obligations.summary
        lines.append(
            "- obligations: "
            f"triggered={obligation_summary.get('triggered', 0)}, "
            f"met={obligation_summary.get('met', 0)}, "
            f"unmet_fail={obligation_summary.get('unmet_fail', 0)}, "
            f"unmet_warn={obligation_summary.get('unmet_warn', 0)}"
        )
    lines.append("")
    if obligations is not None:
        lines.append("Obligations:")
        for entry in obligations.entries:
            check_deadline()
            state = "inactive"
            if entry.get("triggered") is True:
                state = str(entry.get("status") or "unknown")
            lines.append(
                "- "
                f"{entry.get('obligation_id')}: {state} "
                f"({entry.get('enforcement')})"
            )
        lines.append("")
    lines.append("Contradictions:")
    for row in rows:
        check_deadline()
        if row.get("status") != "contradicts":
            continue
        invariant = row.get("invariant", "?")
        path = row.get("path")
        qual = row.get("qual")
        source_kind = row.get("source_row_kind")
        detail = row.get("detail")
        parts = [f"invariant={invariant}"]
        if path:
            parts.append(f"path={path}")
        if qual:
            parts.append(f"qual={qual}")
        if source_kind:
            parts.append(f"row={source_kind}")
        if detail:
            parts.append(f"detail={detail}")
        lines.append(f"- {'; '.join(parts)}")
    if not any(row.get("status") == "contradicts" for row in rows):
        lines.append("- (none)")
    lines.append("")
    lines.append("Proposed invariants (not enforced):")
    for row in rows:
        check_deadline()
        if row.get("status") != "proposed":
            continue
        invariant = row.get("invariant", "?")
        detail = row.get("detail")
        match_count = row.get("match_count")
        would_violate = row.get("would_violate")
        parts = [f"invariant={invariant}"]
        if match_count is not None:
            parts.append(f"matches={match_count}")
        if would_violate is True:
            parts.append("would_violate=true")
        if detail:
            parts.append(f"detail={detail}")
        lines.append(f"- {'; '.join(parts)}")
    if not any(row.get("status") == "proposed" for row in rows):
        lines.append("- (none)")
    lines.append("")
    lines.append("Excess evidence:")
    for row in rows:
        check_deadline()
        if row.get("status") != "excess":
            continue
        evidence_id = row.get("evidence_id", "?")
        evidence_kind = row.get("evidence_kind")
        evidence_display = row.get("evidence_display")
        parts = [f"id={evidence_id}"]
        if evidence_kind:
            parts.append(f"kind={evidence_kind}")
        if evidence_display:
            parts.append(f"display={evidence_display}")
        lines.append(f"- {'; '.join(parts)}")
    if not any(row.get("status") == "excess" for row in rows):
        lines.append("- (none)")
    return lines


def _render_docflow_report_md(doc_id: str, lines: list[str]) -> str:
    frontmatter = [
        "---",
        "doc_revision: 1",
        "reader_reintern: Reader-only: re-intern if doc_revision changed since you last read this doc.",
        f"doc_id: {doc_id}",
        "doc_role: report",
        "doc_scope:",
        "  - repo",
        "  - docflow",
        "  - report",
        "doc_authority: informative",
        "doc_change_protocol: POLICY_SEED.md#change_protocol",
        "doc_requires: []",
        "doc_reviewed_as_of: {}",
        "doc_review_notes: {}",
        "doc_sections:",
        f"  {doc_id}: 1",
        "doc_section_requires:",
        f"  {doc_id}: []",
        "doc_section_reviews:",
        f"  {doc_id}: {{}}",
        "---",
        "",
        f'<a id="{doc_id}"></a>',
        "",
    ]
    return "\n".join(frontmatter + lines) + "\n"


def _emit_docflow_compliance(
    *,
    rows: list[dict[str, object]],
    invariants: Iterable[DocflowInvariant],
    json_output: Path | None,
    md_output: Path | None,
    obligations: DocflowObligationResult | None = None,
) -> None:
    compliance_rows = _docflow_compliance_rows(rows, invariants=invariants)
    if obligations is not None:
        compliance_rows = _decorate_compliance_rows_with_obligations(
            compliance_rows,
            obligations.entries,
        )
    payload = {
        "version": 2,
        "summary": _summarize_docflow_compliance(compliance_rows),
        "rows": compliance_rows,
    }
    if obligations is not None:
        payload["obligations"] = {
            "summary": obligations.summary,
            "entries": obligations.entries,
            "context": obligations.context,
        }
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(
            _render_docflow_report_md(
                "docflow_compliance",
                _render_docflow_compliance_md(
                    compliance_rows,
                    obligations=obligations,
                ),
            )
        )


def _emit_docflow_implication_matrices(
    *,
    docs: dict[str, Doc],
    json_output: Path | None,
) -> None:
    per_doc_lattices: dict[str, dict[str, JSONValue]] = {}
    notions_by_doc: dict[str, list[dict[str, JSONValue]]] = {}
    for rel, payload in docs.items():
        check_deadline()
        notions = _extract_doc_body_notions_with_anchors(path=rel, doc=payload)
        notions_by_doc[rel] = notions
        per_doc_lattices[rel] = _build_doc_implication_lattice(path=rel, notions=notions)
    composed, warnings, conflicts = _compose_doc_dependency_matrices(
        docs=docs,
        per_doc_lattices=per_doc_lattices,
    )
    payload: dict[str, object] = {
        "version": 1,
        "summary": {
            "documents": len(docs),
            "notions": sum(len(items) for items in notions_by_doc.values()),
            "conflicts": len(conflicts),
            "warnings": len(warnings),
        },
        "notions": notions_by_doc,
        "lattices": per_doc_lattices,
        "composed_lattices": composed,
        "warnings": warnings,
        "violations": conflicts,
    }
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_docflow_docs(
    *,
    root: Path,
    extra_paths: list[str] | None,
) -> dict[str, Doc]:
    docs: dict[str, Doc] = {}
    for path in _iter_docflow_paths(root, extra_paths):
        check_deadline()
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        docs[rel] = Doc(frontmatter=fm, body=body)
    return docs


_MANDATORY_HINT_RE = re.compile(r"\b(must|shall|required|never|do not|must not|keep|preserve|use|run)\b", re.IGNORECASE)
_TOGGLE_TOKEN_RE = re.compile(r"`(?P<token>--[a-z0-9-]+|[A-Z][A-Z0-9_]{2,})`")


def _directive_normalized(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[`*_]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _directive_is_mandatory(text: str) -> bool:
    return bool(_MANDATORY_HINT_RE.search(text))


def _directive_is_negative(text: str) -> bool:
    lowered = text.lower()
    return "do not" in lowered or "must not" in lowered or "never" in lowered


def _directive_stem(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(do not|must not|never|must|shall|required)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _agent_scope_root(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return "." if parent in {"", "."} else parent


def _extract_mandatory_directives(path: str, doc: Doc) -> list[AgentDirective]:
    lines = doc.body.splitlines()
    directives: list[AgentDirective] = []
    in_required_behavior = False
    for line_no, raw in enumerate(lines, start=1):
        check_deadline()
        line = raw.strip()
        if line.startswith("## "):
            in_required_behavior = line.lower() == "## required behavior"
            continue
        if not line.startswith("-"):
            continue
        text = line[1:].strip()
        if not text:
            continue
        mandatory = in_required_behavior or _directive_is_mandatory(text)
        directives.append(
            AgentDirective(
                source=path,
                scope_root=_agent_scope_root(path),
                line=line_no,
                text=text,
                normalized=_directive_normalized(text),
                mandatory=mandatory,
                delta_marked=("delta:" in text.lower() or "[delta]" in text.lower()),
            )
        )
    return directives


def _directive_reference_entries(path: str, doc: Doc) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for line_no, raw in enumerate(doc.body.splitlines(), start=1):
        check_deadline()
        match = re.search(r"`([^`]+\.md#[^`]+)`", raw)
        if match:
            refs.append((match.group(1), line_no))
    return refs


def _agent_instruction_graph(
    *,
    root: Path,
    docs: dict[str, Doc],
    json_output: Path,
    md_output: Path,
) -> tuple[list[str], list[str]]:
    required_docs = ["AGENTS.md", "CONTRIBUTING.md", "POLICY_SEED.md"]
    scoped_agent_paths = sorted(path for path in docs if path.endswith("/AGENTS.md"))
    included_docs = [path for path in required_docs if path in docs] + scoped_agent_paths
    directives: list[AgentDirective] = []
    reference_rows: list[dict[str, object]] = []
    for path in included_docs:
        check_deadline()
        doc = docs[path]
        directives.extend(_extract_mandatory_directives(path, doc))
        reference_rows.extend(
            {
                "source": path,
                "reference": ref,
                "line": line,
            }
            for ref, line in _directive_reference_entries(path, doc)
        )

    canonical_directives = [
        directive for directive in directives if directive.source == "AGENTS.md" and directive.mandatory
    ]
    canonical_norms = {directive.normalized for directive in canonical_directives}

    duplicate_mandatory: list[dict[str, object]] = []
    directive_groups: dict[str, list[AgentDirective]] = defaultdict(list)
    for directive in directives:
        check_deadline()
        if directive.mandatory:
            directive_groups[directive.normalized].append(directive)
    for norm, group in sorted(directive_groups.items()):
        check_deadline()
        if len(group) > 1:
            duplicate_mandatory.append(
                {
                    "normalized": norm,
                    "occurrences": [
                        {
                            "source": item.source,
                            "scope_root": item.scope_root,
                            "line": item.line,
                            "text": item.text,
                        }
                        for item in group
                    ],
                }
            )

    precedence_conflicts: list[dict[str, object]] = []
    by_stem: dict[str, list[AgentDirective]] = defaultdict(list)
    for directive in directives:
        check_deadline()
        if directive.mandatory:
            stem = _directive_stem(directive.text)
            if stem:
                by_stem[stem].append(directive)
    for stem, group in sorted(by_stem.items()):
        check_deadline()
        negatives = [item for item in group if _directive_is_negative(item.text)]
        positives = [item for item in group if not _directive_is_negative(item.text)]
        if negatives and positives:
            precedence_conflicts.append(
                {
                    "stem": stem,
                    "positive": [
                        {"source": item.source, "line": item.line, "text": item.text}
                        for item in positives
                    ],
                    "negative": [
                        {"source": item.source, "line": item.line, "text": item.text}
                        for item in negatives
                    ],
                }
            )

    stale_dependency_revisions: dict[str, list[dict[str, object]]] = {}
    revisions: dict[str, int] = {}
    for path, doc in docs.items():
        check_deadline()
        revision = doc.frontmatter.get("doc_revision")
        if isinstance(revision, int):
            revisions[path] = revision
        _add_section_revisions(revisions, rel=path, fm=doc.frontmatter)
    for path in included_docs:
        check_deadline()
        reviewed = docs[path].frontmatter.get("doc_reviewed_as_of", {})
        if not isinstance(reviewed, dict):
            continue
        for dep_ref, pinned in reviewed.items():
            check_deadline()
            if not isinstance(dep_ref, str) or not isinstance(pinned, int):
                continue
            actual = revisions.get(dep_ref)
            if actual is None:
                dep_doc = _doc_ref_base(dep_ref)
                actual = revisions.get(dep_doc)
            if actual is not None and actual != pinned:
                stale_dependency_revisions.setdefault(path, []).append(
                    {
                        "source": path,
                        "dependency": dep_ref,
                        "pinned": pinned,
                        "actual": actual,
                    }
                )

    hidden_operational_toggles: list[dict[str, object]] = []
    agent_toggle_sources = ["AGENTS.md"] + scoped_agent_paths
    agent_toggle_docs = [docs[path] for path in agent_toggle_sources if path in docs]
    visible_tokens = {
        match.group("token")
        for doc in agent_toggle_docs
        for match in _TOGGLE_TOKEN_RE.finditer(doc.body)
    }
    for path in ["POLICY_SEED.md", "CONTRIBUTING.md"]:
        check_deadline()
        if path not in docs:
            continue
        for match in _TOGGLE_TOKEN_RE.finditer(docs[path].body):
            check_deadline()
            token = match.group("token")
            if token not in visible_tokens:
                hidden_operational_toggles.append(
                    {
                        "source": path,
                        "token": token,
                    }
                )

    scoped_delta_violations: list[dict[str, object]] = []
    for directive in directives:
        check_deadline()
        if directive.source == "AGENTS.md" or not directive.mandatory:
            continue
        if directive.source.endswith("/AGENTS.md"):
            if directive.normalized not in canonical_norms and not directive.delta_marked:
                scoped_delta_violations.append(
                    {
                        "source": directive.source,
                        "line": directive.line,
                        "text": directive.text,
                        "reason": "scoped mandatory directive must be canonical or explicitly marked as delta",
                    }
                )

    payload = {
        "root": str(root),
        "included_docs": included_docs,
        "summary": {
            "mandatory_directives": sum(1 for item in directives if item.mandatory),
            "duplicate_mandatory": len(duplicate_mandatory),
            "precedence_conflicts": len(precedence_conflicts),
            "stale_dependency_revisions": sum(
                len(entries) for entries in stale_dependency_revisions.values()
            ),
            "hidden_operational_toggles": len(hidden_operational_toggles),
            "scoped_delta_violations": len(scoped_delta_violations),
        },
        "canonical": {
            "source": "AGENTS.md",
            "directives": [
                {
                    "line": item.line,
                    "text": item.text,
                    "normalized": item.normalized,
                }
                for item in canonical_directives
            ],
        },
        "directive_references": reference_rows,
        "duplicate_mandatory": duplicate_mandatory,
        "precedence_conflicts": precedence_conflicts,
        "stale_dependency_revisions": [
            entry
            for path in sorted(stale_dependency_revisions)
            for entry in stale_dependency_revisions[path]
        ],
        "hidden_operational_toggles": hidden_operational_toggles,
        "scoped_delta_violations": scoped_delta_violations,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Agent Instruction Graph",
        "",
        f"- canonical source: `AGENTS.md`",
        f"- mandatory directives: {payload['summary']['mandatory_directives']}",
        f"- duplicates: {payload['summary']['duplicate_mandatory']}",
        f"- precedence conflicts: {payload['summary']['precedence_conflicts']}",
        f"- stale dependency revisions: {payload['summary']['stale_dependency_revisions']}",
        f"- hidden operational toggles: {payload['summary']['hidden_operational_toggles']}",
        f"- scoped delta violations: {payload['summary']['scoped_delta_violations']}",
        "",
    ]
    if duplicate_mandatory:
        md_lines.extend(["## Duplicate Mandatory Directives", ""])
        for entry in duplicate_mandatory:
            check_deadline()
            md_lines.append(f"- `{entry['normalized']}`")
            for occurrence in entry["occurrences"]:
                md_lines.append(f"  - {occurrence['source']}:{occurrence['line']} — {occurrence['text']}")
        md_lines.append("")
    if precedence_conflicts:
        md_lines.extend(["## Conflicting Precedence", ""])
        for entry in precedence_conflicts:
            check_deadline()
            md_lines.append(f"- stem: `{entry['stem']}`")
            for occurrence in entry["positive"]:
                md_lines.append(f"  - positive: {occurrence['source']}:{occurrence['line']} — {occurrence['text']}")
            for occurrence in entry["negative"]:
                md_lines.append(f"  - negative: {occurrence['source']}:{occurrence['line']} — {occurrence['text']}")
        md_lines.append("")
    if stale_dependency_revisions:
        md_lines.extend(["## Stale Dependency Revisions", ""])
        for path in sorted(stale_dependency_revisions):
            check_deadline()
            for entry in stale_dependency_revisions[path]:
                check_deadline()
                md_lines.append(
                    f"- {entry['source']}: `{entry['dependency']}` pinned `{entry['pinned']}` but current revision is `{entry['actual']}`"
                )
        md_lines.append("")
    if hidden_operational_toggles:
        md_lines.extend(["## Hidden Operational Toggles", ""])
        for entry in hidden_operational_toggles:
            check_deadline()
            md_lines.append(f"- `{entry['token']}` appears in {entry['source']} but not in AGENTS docs")
        md_lines.append("")
    if scoped_delta_violations:
        md_lines.extend(["## Scoped Delta Violations", ""])
        for entry in scoped_delta_violations:
            check_deadline()
            md_lines.append(f"- {entry['source']}:{entry['line']} — {entry['text']}")
        md_lines.append("")
    if not any((duplicate_mandatory, precedence_conflicts, stale_dependency_revisions, hidden_operational_toggles, scoped_delta_violations)):
        md_lines.extend(["No instruction drift detected.", ""])
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text("\n".join(md_lines), encoding="utf-8")

    warnings: list[str] = []
    if hidden_operational_toggles:
        warnings.append(
            "agent instruction graph: hidden operational toggles detected; see "
            f"{md_output.as_posix()}"
        )
    violations: list[str] = []
    if duplicate_mandatory:
        violations.append("agent instruction graph: duplicate mandatory directives detected")
    if precedence_conflicts:
        violations.append("agent instruction graph: conflicting precedence directives detected")
    if stale_dependency_revisions:
        violations.append("agent instruction graph: stale dependency revisions detected")
    if scoped_delta_violations:
        violations.append("agent instruction graph: scoped AGENTS directives must be canonical or explicit deltas")
    return warnings, violations


def _glossary_section_headings(doc: Doc) -> dict[str, str]:
    lines = doc.body.splitlines()
    anchor_re = re.compile(r'^\s*<a id="([^"]+)"></a>\s*$')
    heading_re = re.compile(r"^#{1,6}\s+(.*)$")
    anchors: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        check_deadline()
        match = anchor_re.match(line)
        if match:
            anchors.append((match.group(1), idx))
    headings: dict[str, str] = {}
    for key, idx in anchors:
        check_deadline()
        title = None
        for j in range(idx + 1, len(lines)):
            check_deadline()
            match = heading_re.match(lines[j].strip())
            if match:
                title = match.group(1).strip()
                break
        if title:
            title = re.sub(r"^\d+\.\s*", "", title)
            headings[key] = title
    return headings


def _term_pattern(heading: str) -> re.Pattern[str]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\\s\\-]*", heading):
        return re.compile(rf"\\b{re.escape(heading)}\\b", re.IGNORECASE)
    return re.compile(re.escape(heading), re.IGNORECASE)


def _docflow_canonicality_entries(
    *,
    docs: dict[str, Doc],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    glossary = docs.get("glossary.md")
    if glossary is None:
        return [], [], {"error": "glossary.md not found"}
    sections = glossary.frontmatter.get("doc_sections", {})
    if not isinstance(sections, dict):
        return [], [], {"error": "glossary.md missing doc_sections"}
    headings = _glossary_section_headings(glossary)
    terms = _sorted(str(key) for key in sections.keys())

    entries: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    ambiguous_terms: set[str] = set()
    no_induced_terms: set[str] = set()

    # dataflow-bundle: doc, signal, term
    def _record_signal(term: str, signal: str, doc: str | None = None) -> None:
        signal_rows.append(
            {
                "row_kind": "canonicality_signal",
                "term": term,
                "signal": signal,
                "doc": doc,
            }
        )

    for term in terms:
        check_deadline()
        term_ref = f"glossary.md#{term}"
        heading = headings.get(term)
        pattern = _term_pattern(heading) if heading else None
        requires_docs: set[str] = set()
        explicit_docs: set[str] = set()
        implicit_docs: set[str] = set()

        for rel, doc in docs.items():
            check_deadline()
            if rel == "glossary.md":
                continue
            requires = doc.frontmatter.get("doc_requires", [])
            if isinstance(requires, list) and term_ref in requires:
                requires_docs.add(rel)
            if term_ref in doc.body:
                explicit_docs.add(rel)
            if pattern and rel not in explicit_docs and rel not in requires_docs:
                if pattern.search(doc.body):
                    implicit_docs.add(rel)

        explicit_without_requires = explicit_docs - requires_docs
        requires_without_explicit = requires_docs - explicit_docs
        missing_anchor = term not in headings

        if missing_anchor:
            _record_signal(term, "missing_anchor")
            ambiguous_terms.add(term)
        for rel in _sorted(explicit_without_requires):
            check_deadline()
            _record_signal(term, "explicit_without_requires", rel)
            ambiguous_terms.add(term)
        for rel in _sorted(requires_without_explicit):
            check_deadline()
            _record_signal(term, "requires_without_explicit", rel)
            ambiguous_terms.add(term)
        for rel in _sorted(implicit_docs):
            check_deadline()
            _record_signal(term, "implicit_without_requires", rel)
            ambiguous_terms.add(term)
        if not requires_docs:
            _record_signal(term, "no_induced_meaning")
            no_induced_terms.add(term)

        candidate = (
            term not in ambiguous_terms
            and term not in no_induced_terms
            and bool(requires_docs)
            and not missing_anchor
        )
        entries.append(
            {
                "term": term,
                "heading": heading,
                "anchor_present": not missing_anchor,
                "requires_docs": _sorted(requires_docs),
                "explicit_docs": _sorted(explicit_docs),
                "implicit_docs": _sorted(implicit_docs),
                "explicit_without_requires": _sorted(explicit_without_requires),
                "requires_without_explicit": _sorted(requires_without_explicit),
                "candidate": candidate,
            }
        )

    summary = {
        "total_terms": len(terms),
        "candidates": sum(1 for entry in entries if entry.get("candidate")),
        "ambiguous": len(ambiguous_terms),
        "no_induced_meaning": len(no_induced_terms),
    }
    return entries, signal_rows, summary


def _docflow_dependency_graph(
    docs: dict[str, DocflowDocument],
) -> dict[str, object]:
    nodes: dict[str, dict[str, object]] = {}
    edges: list[dict[str, object]] = []
    for rel, doc in _sorted(docs.items()):
        check_deadline()
        fm = doc.frontmatter
        requires = fm.get("doc_requires", [])
        deps: list[str] = []
        if isinstance(requires, list):
            deps = [
                _doc_ref_base(req)
                for req in requires
                if isinstance(req, str)
            ]
        nodes[rel] = {
            "path": rel,
            "doc_id": fm.get("doc_id"),
            "doc_role": fm.get("doc_role"),
            "doc_authority": fm.get("doc_authority"),
            "doc_dependency_projection": fm.get("doc_dependency_projection"),
            "requires": deps,
        }
        for dep in deps:
            check_deadline()
            edges.append({"from": rel, "to": dep})
    return {"nodes": nodes, "edges": edges}


def _docflow_strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[set[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in graph.get(node, set()):
            check_deadline()
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while True:
                check_deadline()
                popped = stack.pop()
                on_stack.discard(popped)
                component.add(popped)
                if popped == node:
                    break
            components.append(component)

    for node in graph:
        check_deadline()
        if node not in indices:
            visit(node)
    return components


def _docflow_cycles(
    graph: dict[str, JSONValue],
    *,
    lift_roles: set[str] | None = None,
    projection: str | None = None,
) -> list[dict[str, object]]:
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        return []
    lifted = lift_roles or set()
    core = set(CORE_GOVERNANCE_DOCS)
    adjacency: dict[str, set[str]] = {
        key: set() for key in nodes.keys()
    }
    for edge in edges:
        check_deadline()
        if not isinstance(edge, dict):
            continue
        src = edge.get("from")
        dst = edge.get("to")
        if isinstance(src, str) and isinstance(dst, str):
            if lifted:
                meta = nodes.get(src, {})
                if isinstance(meta, dict):
                    role = meta.get("doc_role")
                    if isinstance(role, str) and role in lifted:
                        continue
            if projection == "dependency":
                meta = nodes.get(src, {})
                if isinstance(meta, dict):
                    proj = meta.get("doc_dependency_projection")
                    if isinstance(proj, str):
                        if proj == "glossary_root":
                            continue
                        if proj == "glossary_lifted" and dst in core:
                            continue
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set())
    components = _docflow_strongly_connected_components(adjacency)
    cycles: list[dict[str, object]] = []
    for comp in components:
        check_deadline()
        if not comp:
            continue
        has_self = any(node in adjacency.get(node, set()) for node in comp)
        if len(comp) == 1 and not has_self:
            continue
        ordered = _sorted(comp)
        kind = "non_core"
        if set(ordered).issubset(core):
            kind = "core"
        elif set(ordered) & core:
            kind = "mixed"
        cycles.append(
            {
                "nodes": ordered,
                "kind": kind,
                "size": len(ordered),
            }
        )
    cycles.sort(key=lambda entry: (entry.get("kind"), entry.get("size", 0), entry.get("nodes")))
    return cycles


def _render_docflow_cycles_md(
    *,
    raw_cycles: list[dict[str, object]],
    projection_cycles: list[dict[str, object]],
    projection_label: str,
) -> list[str]:
    lines: list[str] = []
    lines.append("Docflow dependency cycles")
    if not raw_cycles:
        lines.append("- none")
        return lines
    lines.append("Summary (raw graph):")
    counts: dict[str, int] = {}
    for entry in raw_cycles:
        check_deadline()
        kind = str(entry.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    for kind in _sorted(counts):
        check_deadline()
        lines.append(f"- {kind}: {counts[kind]}")
    lines.append("")
    lines.append(f"Summary ({projection_label}):")
    projection_counts: dict[str, int] = {}
    for entry in projection_cycles:
        check_deadline()
        kind = str(entry.get("kind", "unknown"))
        projection_counts[kind] = projection_counts.get(kind, 0) + 1
    if projection_counts:
        for kind in _sorted(projection_counts):
            check_deadline()
            lines.append(f"- {kind}: {projection_counts[kind]}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Cycles (raw graph):")
    for entry in raw_cycles:
        check_deadline()
        nodes = entry.get("nodes", [])
        if not isinstance(nodes, list):
            continue
        kind = entry.get("kind", "unknown")
        lines.append(f"- ({kind}) {', '.join(nodes)}")
    lines.append("")
    lines.append(f"Cycles ({projection_label}):")
    if projection_cycles:
        for entry in projection_cycles:
            check_deadline()
            nodes = entry.get("nodes", [])
            if not isinstance(nodes, list):
                continue
            kind = entry.get("kind", "unknown")
            lines.append(f"- ({kind}) {', '.join(nodes)}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Guidance:")
    lines.append("- core: expected governance cycle; break only with policy change.")
    lines.append("- mixed: consider lifting shared semantics to glossary and removing non-core back-edges.")
    lines.append("- non_core: lift shared semantics to glossary to break the cycle.")
    return lines


def _emit_docflow_cycles(
    docs: dict[str, DocflowDocument],
    *,
    json_output: Path | None,
    md_output: Path | None,
) -> None:
    graph = _docflow_dependency_graph(docs)
    raw_cycles = _docflow_cycles(graph)
    projection_cycles = _docflow_cycles(graph, projection="dependency")
    payload = {
        "graph": graph,
        "cycles": raw_cycles,
        "projection": {
            "mode": "dependency",
            "cycles": projection_cycles,
        },
    }
    if json_output is not None:
        json_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if md_output is not None:
        md_output.write_text(
            _render_docflow_report_md(
                "docflow_cycles",
                _render_docflow_cycles_md(
                    raw_cycles=raw_cycles,
                    projection_cycles=projection_cycles,
                    projection_label="dependency projection",
                ),
            ),
            encoding="utf-8",
        )


_CHANGE_PROTOCOL_CANONICAL = "POLICY_SEED.md#change_protocol"
_CHANGE_PROTOCOL_LEGACY = re.compile(r"^POLICY_SEED\.md\s*§\s*6(?:\b|$)")


def _classify_change_protocol(value: object) -> tuple[str, str | None]:
    if not isinstance(value, str) or not value.strip():
        return "missing", None
    raw = value.strip()
    if raw == _CHANGE_PROTOCOL_CANONICAL:
        return "canonical", raw
    if _CHANGE_PROTOCOL_LEGACY.match(raw):
        return "legacy", _CHANGE_PROTOCOL_CANONICAL
    return "custom", raw


def _docflow_change_protocol_entries(
    docs: dict[str, Doc],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for rel in _sorted(docs):
        check_deadline()
        fm = docs[rel].frontmatter
        status, normalized = _classify_change_protocol(fm.get("doc_change_protocol"))
        entries.append(
            {
                "path": rel,
                "doc_id": fm.get("doc_id") if isinstance(fm.get("doc_id"), str) else None,
                "raw": fm.get("doc_change_protocol"),
                "status": status,
                "normalized": normalized,
            }
        )
    return entries


def _render_docflow_change_protocol_md(
    entries: list[dict[str, object]],
) -> list[str]:
    counts: Counter[str] = Counter()
    by_status: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        check_deadline()
        status = str(entry.get("status") or "unknown")
        counts[status] += 1
        by_status[status].append(entry)
    lines: list[str] = ["Docflow change-protocol normalization report"]
    lines.append("Summary:")
    for status in _sorted(counts):
        check_deadline()
        lines.append(f"- {status}: {counts[status]}")
    for status in ("legacy", "custom", "missing"):
        check_deadline()
        rows = by_status.get(status) or []
        if not rows:
            continue
        lines.append(f"{status.capitalize()} entries:")
        for entry in rows:
            check_deadline()
            path = entry.get("path") or "?"
            raw = entry.get("raw")
            normalized = entry.get("normalized")
            suffix = ""
            if normalized and normalized != raw:
                suffix = f" -> {normalized}"
            lines.append(f"- {path}: {raw}{suffix}")
    return lines


def _emit_docflow_change_protocol(
    docs: dict[str, Doc],
    *,
    json_output: Path,
    md_output: Path,
) -> None:
    entries = _docflow_change_protocol_entries(docs)
    payload = {
        "canonical": _CHANGE_PROTOCOL_CANONICAL,
        "entries": entries,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(
        _render_docflow_report_md(
            "docflow_change_protocol",
            _render_docflow_change_protocol_md(entries),
        ),
        encoding="utf-8",
    )


def _docflow_section_review_rows(
    docs: dict[str, Doc],
    revisions: dict[str, int],
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    for rel in _sorted(docs):
        check_deadline()
        fm = docs[rel].frontmatter
        doc_id = fm.get("doc_id") if isinstance(fm.get("doc_id"), str) else None
        base = _docflow_base_meta(rel, doc_id)
        sections = fm.get("doc_sections")
        requires_map = fm.get("doc_section_requires")
        reviews_map = fm.get("doc_section_reviews")
        if not isinstance(sections, dict) or not isinstance(requires_map, dict):
            continue
        review_lookup = reviews_map if isinstance(reviews_map, dict) else {}
        for anchor, deps in requires_map.items():
            check_deadline()
            if not isinstance(anchor, str) or not anchor:
                continue
            anchor_version = sections.get(anchor)
            if anchor_version is None:
                warnings.append(f"{rel}: doc_section_requires references unknown anchor {anchor}")
                continue
            if not isinstance(deps, list):
                warnings.append(f"{rel}: doc_section_requires.{anchor} must be a list")
                continue
            anchor_reviews = review_lookup.get(anchor, {})
            if anchor not in review_lookup:
                warnings.append(f"{rel}: missing doc_section_reviews for anchor {anchor}")
            if anchor in review_lookup and not isinstance(anchor_reviews, dict):
                warnings.append(f"{rel}: doc_section_reviews.{anchor} must be a map")
                anchor_reviews = {}
            for dep in deps:
                check_deadline()
                if not isinstance(dep, str) or not dep:
                    continue
                expected_dep_version = revisions.get(dep)
                review_entry = anchor_reviews.get(dep) if isinstance(anchor_reviews, dict) else None
                status = "ok"
                dep_version = None
                self_version = None
                outcome = None
                note = None
                if expected_dep_version is None:
                    status = "unknown_dependency"
                if not isinstance(review_entry, dict):
                    status = "missing_review"
                else:
                    dep_version = review_entry.get("dep_version")
                    self_version = review_entry.get("self_version_at_review")
                    outcome = review_entry.get("outcome")
                    note = review_entry.get("note")
                    if not isinstance(dep_version, int) or expected_dep_version is None:
                        status = "invalid_dep_version"
                    elif dep_version != expected_dep_version:
                        status = "stale_dep"
                    if not isinstance(self_version, int):
                        status = "invalid_self_version"
                    elif isinstance(anchor_version, int) and self_version != anchor_version:
                        status = "stale_self"
                    if outcome not in {"no_change", "changed"}:
                        status = "invalid_outcome"
                    if not isinstance(note, str) or not note.strip():
                        status = "missing_note"
                if status != "ok":
                    warnings.append(
                        f"{rel}: {anchor} review for {dep} status={status}"
                    )
                rows.append(
                    {
                        "row_kind": "doc_section_review",
                        **base,
                        "anchor": anchor,
                        "anchor_version": anchor_version,
                        "dep": dep,
                        "expected_dep_version": expected_dep_version,
                        "dep_version": dep_version,
                        "self_version_at_review": self_version,
                        "outcome": outcome,
                        "note_present": isinstance(note, str) and bool(note.strip()),
                        "status": status,
                    }
                )
    return rows, warnings


def _render_docflow_section_reviews_md(
    rows: list[dict[str, object]],
) -> list[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        check_deadline()
        status = str(row.get("status") or "unknown")
        counts[status] += 1
    lines: list[str] = ["Docflow anchor review report"]
    lines.append("Summary:")
    for status in _sorted(counts):
        check_deadline()
        lines.append(f"- {status}: {counts[status]}")
    for row in rows:
        check_deadline()
        status = str(row.get("status") or "unknown")
        if status == "ok":
            continue
        path = row.get("path") or "?"
        anchor = row.get("anchor") or "?"
        dep = row.get("dep") or "?"
        expected = row.get("expected_dep_version")
        seen = row.get("dep_version")
        self_version = row.get("self_version_at_review")
        lines.append(
            f"- {path}::{anchor} -> {dep} status={status} dep={seen}/{expected} self={self_version}"
        )
    return lines


def _emit_docflow_section_reviews(
    docs: dict[str, Doc],
    revisions: dict[str, int],
    *,
    json_output: Path,
    md_output: Path,
) -> None:
    rows, _ = _docflow_section_review_rows(docs, revisions)
    payload = {"rows": rows}
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(
        _render_docflow_report_md(
            "docflow_section_reviews",
            _render_docflow_section_reviews_md(rows),
        ),
        encoding="utf-8",
    )

def _canonicality_predicates() -> dict[str, Callable[[Mapping[str, JSONValue], Mapping[str, JSONValue]], bool]]:
    def _is_ambiguous(row: Mapping[str, JSONValue], _: Mapping[str, JSONValue]) -> bool:
        if row.get("row_kind") != "canonicality_signal":
            return False
        signal = str(row.get("signal") or "")
        return signal in {
            "missing_anchor",
            "explicit_without_requires",
            "requires_without_explicit",
            "implicit_without_requires",
        }

    return {"canonicality_is_ambiguous": _is_ambiguous}


@cache
def _docflow_canonicality_spec() -> ProjectionSpec:
    return ProjectionSpec(
        spec_version=1,
        name="docflow_canonicality_ambiguity",
        domain="docflow_canonicality",
        pipeline=(
            ProjectionOp("select", {"predicates": ["canonicality_is_ambiguous"]}),
            ProjectionOp("project", {"fields": ["term", "signal", "doc"]}),
            ProjectionOp("count_by", {"fields": ["term"]}),
            ProjectionOp("sort", {"by": ["term"]}),
        ),
    )


@decision_protocol
@cache
def _docflow_canonicality_execution_ops():
    return execution_ops_from_spec(_docflow_canonicality_spec())


def _render_docflow_canonicality_md(
    entries: list[dict[str, object]],
    summary: dict[str, object],
    *,
    convergence: dict[str, object],
    spec_id: str,
) -> list[str]:
    lines: list[str] = []
    lines.append("Docflow canonicality report")
    lines.append(f"- total_terms: {summary.get('total_terms', 0)}")
    lines.append(f"- candidates: {summary.get('candidates', 0)}")
    lines.append(f"- ambiguous: {summary.get('ambiguous', 0)}")
    lines.append(f"- no_induced_meaning: {summary.get('no_induced_meaning', 0)}")
    lines.append(f"- projection_spec_id: {spec_id}")
    lines.append("")

    candidates = [entry for entry in entries if entry.get("candidate")]
    lines.append("Canonicality candidates:")
    if not candidates:
        lines.append("- (none)")
    else:
        for entry in _sorted(candidates, key=lambda e: str(e.get("term"))):
            check_deadline()
            heading = entry.get("heading") or ""
            requires = entry.get("requires_docs") or []
            lines.append(
                f"- {entry.get('term')} ({heading}) requires={len(requires)}"
            )
    lines.append("")

    ambiguous = [
        entry
        for entry in entries
        if not entry.get("candidate")
        and (
            entry.get("explicit_without_requires")
            or entry.get("requires_without_explicit")
            or entry.get("implicit_docs")
            or not entry.get("anchor_present", True)
        )
    ]
    lines.append("Ambiguity signals:")
    if not ambiguous:
        lines.append("- (none)")
    else:
        for entry in _sorted(ambiguous, key=lambda e: str(e.get("term"))):
            check_deadline()
            term = entry.get("term")
            reasons: list[str] = []
            if not entry.get("anchor_present", True):
                reasons.append("missing_anchor")
            if entry.get("explicit_without_requires"):
                reasons.append("explicit_without_requires")
            if entry.get("requires_without_explicit"):
                reasons.append("requires_without_explicit")
            if entry.get("implicit_docs"):
                reasons.append("implicit_without_requires")
            reason_blob = ", ".join(reasons) if reasons else "unknown"
            lines.append(f"- {term}: {reason_blob}")
    lines.append("")

    no_induced = [entry for entry in entries if not entry.get("requires_docs")]
    lines.append("No induced meaning (no doc_requires references):")
    if not no_induced:
        lines.append("- (none)")
    else:
        for entry in _sorted(no_induced, key=lambda e: str(e.get("term"))):
            check_deadline()
            heading = entry.get("heading") or ""
            lines.append(f"- {entry.get('term')} ({heading})")
    lines.append("")

    lines.append("Convergence (docflow vs projection spec):")
    lines.append(f"- matched: {convergence.get('matched', False)}")
    docflow_terms = convergence.get("docflow_terms", [])
    projection_terms = convergence.get("projection_terms", [])
    lines.append(f"- docflow_ambiguous_terms: {len(docflow_terms)}")
    lines.append(f"- projection_ambiguous_terms: {len(projection_terms)}")
    if not convergence.get("matched", False):
        lines.append(f"- docflow_only: {convergence.get('docflow_only', [])}")
        lines.append(f"- projection_only: {convergence.get('projection_only', [])}")
    return lines


def _emit_docflow_canonicality(
    *,
    root: Path,
    extra_paths: list[str] | None,
    json_output: Path | None,
    md_output: Path | None,
) -> None:
    docs = _load_docflow_docs(root=root, extra_paths=extra_paths)
    entries, signal_rows, summary = _docflow_canonicality_entries(docs=docs)

    spec = _docflow_canonicality_spec()
    op_registry = _canonicality_predicates()
    projection_rows = apply_execution_ops(
        _docflow_canonicality_execution_ops(),
        signal_rows,
        op_registry=op_registry,
    )
    projection_terms = {row.get("term") for row in projection_rows if row.get("term")}
    docflow_terms = {
        entry.get("term")
        for entry in entries
        if not entry.get("candidate")
        and (
            entry.get("explicit_without_requires")
            or entry.get("requires_without_explicit")
            or entry.get("implicit_docs")
            or not entry.get("anchor_present", True)
        )
    }
    projection_terms = {str(term) for term in projection_terms if term}
    docflow_terms = {str(term) for term in docflow_terms if term}
    convergence = {
        "matched": docflow_terms == projection_terms,
        "docflow_terms": _sorted(docflow_terms),
        "projection_terms": _sorted(projection_terms),
        "docflow_only": _sorted(docflow_terms - projection_terms),
        "projection_only": _sorted(projection_terms - docflow_terms),
    }

    payload = {
        "summary": summary,
        "spec": {
            "id": spec_hash(spec),
            "payload": json.loads(spec_canonical_json(spec)),
        },
        "entries": entries,
        "signals": signal_rows,
        "projection_summary": projection_rows,
        "convergence": convergence,
    }
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if md_output is not None:
        md_output.parent.mkdir(parents=True, exist_ok=True)
        md_output.write_text(
            _render_docflow_report_md(
                "docflow_canonicality",
                _render_docflow_canonicality_md(
                    entries,
                    summary,
                    convergence=convergence,
                    spec_id=payload["spec"]["id"],
                ),
            ),
            encoding="utf-8",
        )


def _evaluate_docflow_invariants(
    rows: list[dict[str, object]],
    *,
    invariants: Iterable[DocflowInvariant],
) -> list[str]:
    violations: list[str] = []
    op_registry = _docflow_predicates()
    non_evidence_rows = [row for row in rows if row.get("row_kind") != "evidence_key"]
    for invariant in invariants:
        check_deadline()
        rows_to_match = rows if _invariant_uses_evidence_rows(invariant) else non_evidence_rows
        matched = _match_docflow_rows(
            rows_to_match,
            matcher=invariant.matcher,
            op_registry=op_registry,
        )
        if invariant.kind == "never":
            for row in matched:
                check_deadline()
                violations.append(_format_docflow_violation(row))
        elif invariant.kind == "require":
            if not matched:
                violations.append(f"docflow invariant failed: {invariant.name}")
    return violations


def _parse_docflow_invariant_entry(entry: object) -> DocflowInvariant | None:
    if isinstance(entry, dict):
        kind_raw = str(entry.get("kind", "never") or "never").strip().lower()
        if kind_raw not in {"cover", "never", "require"}:
            return None
        kind = cast(Literal["cover", "never", "require"], kind_raw)
        status_raw = str(entry.get("status", "active") or "active").strip().lower()
        status = status_raw if status_raw in {"active", "proposed"} else "active"
        cover_kind = entry.get("cover_evidence_kind")
        cover_id = entry.get("cover_evidence_id")
        cover_source = entry.get("cover_evidence_source")
        if cover_kind or cover_id or cover_source:
            predicates: list[str] = []
            params: dict[str, JSONValue] = {}

            def _collect_list(value: object) -> list[str]:
                if isinstance(value, list):
                    return [str(v).strip() for v in value if str(v).strip()]
                if isinstance(value, str) and value.strip():
                    return [value.strip()]
                return []

            kind_list = _collect_list(cover_kind)
            if kind_list:
                predicates.append("evidence_kind")
                params["evidence_kinds" if len(kind_list) > 1 else "evidence_kind"] = (
                    kind_list if len(kind_list) > 1 else kind_list[0]
                )
            id_list = _collect_list(cover_id)
            if id_list:
                predicates.append("evidence_id")
                params["evidence_ids" if len(id_list) > 1 else "evidence_id"] = (
                    id_list if len(id_list) > 1 else id_list[0]
                )
            source_list = _collect_list(cover_source)
            if source_list:
                predicates.append("evidence_source")
                params["evidence_sources" if len(source_list) > 1 else "evidence_source"] = (
                    source_list if len(source_list) > 1 else source_list[0]
                )
            if predicates:
                name = str(entry.get("name") or "docflow:cover")
                matcher = _parse_docflow_predicate_matcher(
                    predicates=predicates,
                    params=params,
                )
                if matcher is None:
                    return None
                return DocflowInvariant(name=name, kind="cover", matcher=matcher, status=status)
        spec_payload = entry.get("spec")
        spec_json = entry.get("spec_json")
        spec_data: dict[str, JSONValue] | None = None
        if isinstance(spec_payload, dict):
            spec_data = {str(k): spec_payload[k] for k in spec_payload}
        elif isinstance(spec_json, str) and spec_json.strip():
            try:
                spec_data = json.loads(spec_json)
            except Exception:
                spec_data = None
        if spec_data is None:
            return None
        spec = spec_from_dict(spec_data)
        matcher = _matcher_from_spec(spec)
        if matcher is None:
            return None
        name = str(entry.get("name") or spec.name or "docflow:custom")
        return DocflowInvariant(name=name, kind=kind, matcher=matcher, status=status)
    if isinstance(entry, str):
        entry = entry.strip()
        if entry.startswith("{") and entry.endswith("}"):
            try:
                spec_data = json.loads(entry)
            except Exception:
                return None
            spec = spec_from_dict(spec_data)
            matcher = _matcher_from_spec(spec)
            if matcher is None:
                return None
            return DocflowInvariant(
                name=str(spec.name or "docflow:inline"),
                kind="never",
                matcher=matcher,
            )
    return None


def _parse_inline_docflow_invariants(rel: str, body: str) -> list[DocflowInvariant]:
    invariants: list[DocflowInvariant] = []
    pattern = re.compile(r"docflow:\s*(never|require)\(([^)]*)\)")
    for lineno, line in enumerate(body.splitlines(), start=1):
        check_deadline()
        for match in pattern.finditer(line):
            check_deadline()
            kind = match.group(1).strip().lower()
            raw_pred = match.group(2).strip()
            if not raw_pred:
                continue
            predicates = [part.strip() for part in raw_pred.split(",") if part.strip()]
            name = f"docflow:{rel}:{lineno}:{kind}"
            matcher = _parse_docflow_predicate_matcher(predicates=predicates)
            if matcher is None:
                continue
            invariants.append(
                DocflowInvariant(
                    name=name,
                    kind=cast(Literal["never", "require"], kind),
                    matcher=matcher,
                    status="active",
                )
            )
    return invariants


def _collect_docflow_invariants(
    docs: dict[str, Doc],
) -> list[DocflowInvariant]:
    invariants: list[DocflowInvariant] = list(DOCFLOW_AUDIT_INVARIANTS)
    for rel, payload in docs.items():
        check_deadline()
        fm = payload.frontmatter
        inv_list = fm.get("doc_invariants")
        if isinstance(inv_list, list):
            for entry in inv_list:
                check_deadline()
                custom = _parse_docflow_invariant_entry(entry)
                if custom is not None:
                    invariants.append(custom)
        invariants.extend(_parse_inline_docflow_invariants(rel, payload.body))
    return invariants


def _lower_name(path: Path) -> str:
    return path.stem.lower()


def _docflow_audit_context(
    root: Path,
    extra_paths: list[str] | None = None,
    *,
    extra_strict: bool = False,
    sppf_gh_ref_mode: SppfGhRefMode = "required",
) -> DocflowAuditContext:
    violations: List[str] = []
    warnings: List[str] = []
    frontmatter_parse_warnings_seen: set[str] = set()

    docs: dict[str, Doc] = {}
    doc_ids: dict[str, str] = {}
    extra_revisions: dict[str, int] = {}
    skipped_no_frontmatter: set[str] = set()
    missing_frontmatter: set[str] = set()

    def _record_frontmatter_parse_warning(
        *,
        rel: str,
        mode: str,
        detail: str | None,
    ) -> None:
        message = _frontmatter_parse_warning(path=rel, mode=mode, detail=detail)
        if not message:
            return
        if message in frontmatter_parse_warnings_seen:
            return
        frontmatter_parse_warnings_seen.add(message)
        warnings.append(message)

    def _load_doc(path: Path, rel: str, *, strict: bool = False) -> None:
        if rel in docs:
            return
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        fm, body, mode, detail = _parse_frontmatter_with_mode(text)
        _record_frontmatter_parse_warning(rel=rel, mode=mode, detail=detail)
        if not fm:
            if strict:
                missing_frontmatter.add(rel)
            else:
                warnings.append(f"{rel}: missing frontmatter; skipping")
            skipped_no_frontmatter.add(rel)
            return
        docs[rel] = Doc(frontmatter=fm, body=body)
        doc_id = fm.get("doc_id")
        if isinstance(doc_id, str):
            if doc_id in doc_ids:
                violations.append(
                    f"duplicate doc_id '{doc_id}' in {rel} and {doc_ids[doc_id]}"
                )
            else:
                doc_ids[doc_id] = rel

    def _iter_extra_docs(paths: list[str]) -> list[Path]:
        extra: list[Path] = []
        for entry in paths:
            check_deadline()
            if not entry:
                continue
            raw_path = Path(entry)
            path = raw_path if raw_path.is_absolute() else root / raw_path
            if path.is_dir():
                extra.extend(_sorted(path.rglob("*.md")))
            elif path.is_file():
                extra.append(path)
        return extra

    def _load_extra_revision(path: Path, rel: str) -> None:
        if rel in docs or rel in extra_revisions or rel in skipped_no_frontmatter:
            return
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        fm, _, mode, detail = _parse_frontmatter_with_mode(text)
        _record_frontmatter_parse_warning(rel=rel, mode=mode, detail=detail)
        if not fm:
            skipped_no_frontmatter.add(rel)
            return
        doc_id = fm.get("doc_id")
        revision = fm.get("doc_revision")
        if not isinstance(doc_id, str) or not isinstance(revision, int):
            skipped_no_frontmatter.add(rel)
            return
        extra_revisions[rel] = revision
        _add_section_revisions(extra_revisions, rel=rel, fm=fm)

    default_relpaths = _iter_default_docflow_relpaths(root)
    for rel in default_relpaths:
        check_deadline()
        path = root / rel
        if not path.exists():
            if rel in GOVERNANCE_DOCS:
                violations.append(f"missing governance doc: {rel}")
            continue
        _load_doc(path, rel)

    if extra_paths:
        for path in _iter_extra_docs(extra_paths):
            check_deadline()
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.as_posix()
            if extra_strict:
                _load_doc(path, rel, strict=True)
            else:
                _load_extra_revision(path, rel)

    implication_docs: dict[str, Doc] = dict(docs)
    for rel in [*_iter_in_governance_relpaths(root), *_iter_out_governance_relpaths(root)]:
        check_deadline()
        path = root / rel
        _load_extra_revision(path, rel)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        fm, body, mode, detail = _parse_frontmatter_with_mode(text)
        _record_frontmatter_parse_warning(rel=rel, mode=mode, detail=detail)
        if not fm:
            warnings.append(f"{rel}: missing frontmatter; implication matrix skipped")
            continue
        implication_docs[rel] = Doc(frontmatter=fm, body=body)

    governance_set = set(GOVERNANCE_DOCS)
    core_set = set(CORE_GOVERNANCE_DOCS)

    revisions: dict[str, int] = {}
    for rel, payload in docs.items():
        check_deadline()
        fm = payload.frontmatter
        if isinstance(fm.get("doc_revision"), int):
            revisions[rel] = fm["doc_revision"]
        _add_section_revisions(revisions, rel=rel, fm=fm)
    revisions.update(extra_revisions)

    invariant_rows, invariant_warnings = _docflow_invariant_rows(
        docs=docs,
        revisions=revisions,
        core_set=core_set,
        missing_frontmatter=missing_frontmatter,
        implication_docs=implication_docs,
    )
    evidence_payload = _load_test_evidence(root)
    if evidence_payload is not None:
        invariant_rows.extend(_evidence_rows_from_test_evidence(evidence_payload))
    else:
        warnings.append("docflow: missing test_evidence.json; evidence compliance skipped")
    invariant_rows.extend(_impact_rows(root))
    warnings.extend(invariant_warnings)
    invariants = _collect_docflow_invariants(docs)
    violations.extend(_evaluate_docflow_invariants(invariant_rows, invariants=invariants))

    warnings.extend(_tooling_warnings(root, docs))
    warnings.extend(_influence_warnings(root))
    sppf_sync_violations, sppf_sync_warnings = _sppf_sync_check(root, mode=sppf_gh_ref_mode)
    violations.extend(sppf_sync_violations)
    warnings.extend(sppf_sync_warnings)
    violations.extend(_sppf_status_triplet_violations(root))
    sppf_violations, sppf_warnings = _sppf_axis_audit(root, docs)
    violations.extend(sppf_violations)
    warnings.extend(sppf_warnings)
    _, section_review_warnings = _docflow_section_review_rows(docs, revisions)
    warnings.extend(section_review_warnings)

    _ = governance_set
    return DocflowAuditContext(
        docs=docs,
        revisions=revisions,
        invariant_rows=invariant_rows,
        invariants=invariants,
        warnings=warnings,
        violations=violations,
    )


def _docflow_audit(
    root: Path,
    extra_paths: list[str] | None = None,
    *,
    extra_strict: bool = False,
    sppf_gh_ref_mode: SppfGhRefMode = "required",
) -> Tuple[List[str], List[str]]:
    context = _docflow_audit_context(
        root,
        extra_paths=extra_paths,
        extra_strict=extra_strict,
        sppf_gh_ref_mode=sppf_gh_ref_mode,
    )
    return context.violations, context.warnings


def _tooling_warnings(root: Path, docs: dict[str, Doc]) -> List[str]:
    warnings: List[str] = []
    makefile = root / "Makefile"
    if makefile.exists():
        for rel in ("README.md", "CONTRIBUTING.md"):
            check_deadline()
            doc = docs.get(rel)
            body = doc.body if doc is not None else ""
            if "make " not in body and "Make targets" not in body:
                warnings.append(
                    f"{rel}: Makefile present but make targets are not documented"
                )
    checks_script = root / "scripts" / "checks.sh"
    if checks_script.exists():
        doc = docs.get("CONTRIBUTING.md")
        body = doc.body if doc is not None else ""
        if "gabion checks" not in body:
            warnings.append(
                "CONTRIBUTING.md: gabion checks present via scripts/checks.sh wrapper but not documented"
            )
    return warnings


def _influence_warnings(root: Path) -> List[str]:
    warnings: List[str] = []
    inbox = GOVERNANCE_PATHS.in_dir(root=root)
    index_path = GOVERNANCE_PATHS.influence_index_path(root=root)
    if not inbox.exists():
        return warnings
    if not index_path.exists():
        warnings.append("docs/influence_index.md: missing influence index for in/")
        return warnings
    index_text = index_path.read_text(encoding="utf-8")
    for path in _sorted(inbox.glob("in-*.md")):
        check_deadline()
        rel = path.as_posix()
        if rel not in index_text:
            warnings.append(f"docs/influence_index.md: missing {rel}")
    return warnings


def _git_diff_paths(rev_range: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", rev_range],
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


SppfGhRefMode: TypeAlias = Literal["advisory", "required"]


def _resolve_sppf_gh_ref_mode(raw: str | None) -> SppfGhRefMode:
    value = (raw or "").strip().lower()
    if value in {"", "required"}:
        return "required"
    if value == "advisory":
        return "advisory"
    raise ValueError(f"invalid SPPF GH-reference mode: {raw!r}")


def _load_sppf_sync_module():
    from gabion.tooling.sppf import sync_core
    return sync_core


def _sppf_sync_check(
    root: Path,
    *,
    mode: SppfGhRefMode,
    load_sppf_sync_module_fn=_load_sppf_sync_module,
    git_diff_paths_fn=_git_diff_paths,
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: List[str] = []
    try:
        sppf_sync = load_sppf_sync_module_fn()
    except Exception:
        return violations, warnings

    try:
        rev_range = sppf_sync._default_range()
    except Exception:
        return violations, warnings

    changed = git_diff_paths_fn(rev_range)
    if not changed:
        return violations, warnings

    relevant = [
        path
        for path in changed
        if GOVERNANCE_PATHS.is_sppf_relevant_path(path)
    ]
    if not relevant:
        return violations, warnings

    try:
        commits = sppf_sync._collect_commits(rev_range)
    except Exception:
        return violations, warnings
    if not commits:
        return violations, warnings

    issue_ids = sppf_sync._issue_ids_from_commits(commits)
    if issue_ids:
        return violations, warnings

    sample = ", ".join(_sorted(relevant)[:5])
    suffix = f" (e.g. {sample})" if sample else ""
    message = (
        f"sppf_sync: no GH references found in commit range {rev_range} touching "
        f"SPPF-relevant paths{suffix}"
    )
    if mode == "required":
        violations.append(message)
    else:
        warnings.append(message)
    return violations, warnings


def _doc_status_changed(paths: Iterable[str]) -> bool:
    tracked_docs = {"docs/sppf_checklist.md", "docs/influence_index.md"}
    for path in paths:
        check_deadline()
        if path in tracked_docs:
            return True
        if path.startswith("in/") and path.endswith(".md"):
            return True
    return False


def _has_doc_status_consistency_violations(violations: Iterable[str]) -> bool:
    for entry in violations:
        check_deadline()
        if "docs/sppf_checklist.md: doc status" in entry:
            return True
        if "missing in docs/influence_index.md" in entry:
            return True
    return False


def _evaluate_docflow_obligations(
    *,
    root: Path,
    violations: list[str],
    baseline_write_emitted: bool,
    delta_guard_checked: bool,
) -> DocflowObligationResult:
    rev_range = "HEAD~1..HEAD"
    try:
        from gabion.tooling.sppf import sync_core
        rev_range = sync_core._default_range()
    except Exception:
        pass

    changed = _git_diff_paths(rev_range)
    relevant_prefixes = ("src/", "in/")
    relevant_paths = {"docs/sppf_checklist.md"}
    sppf_relevant_changed = any(
        path in relevant_paths or any(path.startswith(prefix) for prefix in relevant_prefixes)
        for path in changed
    )
    gh_reference_validated = True
    commit_payloads: list[dict[str, JSONValue]] = []
    issue_ids_payload: list[str] = []
    checklist_impact_payload: list[dict[str, JSONValue]] = []
    issue_lifecycle_payloads: list[dict[str, JSONValue]] = []
    issue_lifecycle_errors: list[str] = []
    issue_lifecycle_fetch_status = "not_applicable"
    if sppf_relevant_changed:
        gh_reference_validated = False
        try:
            from gabion.tooling.sppf import sync_core
            commits = sync_core._collect_commits(rev_range)
            issue_link = sync_core._build_issue_link_facet(commits)
            commit_payloads = [
                {
                    "sha": commit.sha,
                    "subject": commit.subject,
                }
                for commit in commits
            ]
            issue_ids_payload = list(issue_link.issue_ids)
            checklist_impact_payload = [
                {
                    "issue_id": issue_id,
                    "commit_count": count,
                }
                for issue_id, count in issue_link.checklist_impact
            ]
            gh_reference_validated = bool(issue_link.issue_ids)
            if issue_ids_payload:
                issue_lifecycle_fetch_status = "ok"
                for issue_id in issue_ids_payload:
                    try:
                        lifecycle = sync_core._fetch_issue(issue_id)
                    except Exception as exc:
                        issue_lifecycle_errors.append(str(exc))
                        continue
                    issue_lifecycle_payloads.append(
                        {
                            "issue_id": lifecycle.issue_id,
                            "state": lifecycle.state,
                            "labels": list(lifecycle.labels),
                            "url": lifecycle.url,
                        }
                    )
                if issue_lifecycle_errors:
                    issue_lifecycle_fetch_status = (
                        "partial_error" if issue_lifecycle_payloads else "error"
                    )
        except Exception:
            gh_reference_validated = False

    doc_status_changed = _doc_status_changed(changed)
    checklist_influence_consistent = not _has_doc_status_consistency_violations(violations)
    context: dict[str, JSONValue] = {
        "repo_root": str(root),
        "changed_paths": changed,
        "sppf_relevant_paths_changed": sppf_relevant_changed,
        "gh_reference_validated": gh_reference_validated,
        "baseline_write_emitted": baseline_write_emitted,
        "delta_guard_checked": delta_guard_checked,
        "doc_status_changed": doc_status_changed,
        "checklist_influence_consistent": checklist_influence_consistent,
        "rev_range": rev_range,
        "commits": commit_payloads,
        "issue_ids": issue_ids_payload,
        "checklist_impact": checklist_impact_payload,
        "issue_lifecycle_fetch_status": issue_lifecycle_fetch_status,
        "issue_lifecycles": issue_lifecycle_payloads,
        "issue_lifecycle_errors": issue_lifecycle_errors,
    }
    entries = evaluate_obligations(operation="docflow_plan", context=context)
    summary = summarize_obligations(entries)
    obligation_warnings: list[str] = []
    obligation_violations: list[str] = []
    for entry in entries:
        check_deadline()
        if entry.get("triggered") is not True or entry.get("status") != "unmet":
            continue
        message = (
            f"obligation unmet [{entry.get('obligation_id')}]: "
            f"{entry.get('description')}"
        )
        if entry.get("enforcement") == "fail":
            obligation_violations.append(message)
        else:
            obligation_warnings.append(message)
    return DocflowObligationResult(
        entries=entries,
        summary=summary,
        warnings=obligation_warnings,
        violations=obligation_violations,
        context=context,
    )


def _decorate_compliance_rows_with_obligations(
    rows: list[dict[str, object]],
    obligations: list[dict[str, JSONValue]],
) -> list[dict[str, object]]:
    active_ids = [
        str(entry.get("obligation_id") or "")
        for entry in obligations
        if entry.get("triggered") is True
    ]
    if not active_ids:
        return rows
    decorated: list[dict[str, object]] = []
    for row in rows:
        check_deadline()
        item = dict(row)
        item["obligations"] = list(active_ids)
        decorated.append(item)
    return decorated


# --- Decision tier candidate helpers ---


def _decision_tier_candidates(lint_path: Path, *, tier: int, output_format: str) -> int:
    codes = {"GABION_DECISION_SURFACE", "GABION_VALUE_DECISION_SURFACE"}
    keys: list[str] = []
    for line in lint_path.read_text().splitlines():
        check_deadline()
        parsed = _parse_lint_entry(line)
        if parsed is None:
            continue
        if parsed.code not in codes:
            continue
        keys.append(f"{parsed.path}:{parsed.line}:{parsed.col}")

    keys = _sorted(set(keys))
    if output_format == "lines":
        for key in keys:
            check_deadline()
            print(key)
        return 0

    tier_key = f"tier{tier}"
    print("[decision]")
    print(f"{tier_key} = [")
    for key in keys:
        check_deadline()
        print(f"  \"{key}\",")
    print("]")
    return 0


# --- Consolidation audit helpers ---


def _parse_surfaces(lines: Iterable[str], *, value_encoded: bool) -> list[DecisionSurface]:
    surfaces: list[DecisionSurface] = []
    for line in lines:
        check_deadline()
        outcome = parse_surface_line(line, value_encoded=value_encoded)
        if outcome is None:
            continue
        surfaces.append(
            DecisionSurface(
                path=outcome.path,
                qual=outcome.qual,
                params=outcome.params,
                meta=outcome.meta,
            )
        )
    return surfaces


def _surfaces_from_forest(
    forest: dict[str, JSONValue],
) -> tuple[list[DecisionSurface], list[DecisionSurface]]:
    nodes = forest.get("nodes")
    alts = forest.get("alts")
    if not isinstance(nodes, list) or not isinstance(alts, list):
        return [], []

    node_meta: dict[tuple[str, tuple[object, ...]], dict[str, object]] = {}
    for node in nodes:
        check_deadline()
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        key = node.get("key")
        if not isinstance(kind, str) or not isinstance(key, list):
            continue
        meta = node.get("meta")
        node_meta[(kind, tuple(key))] = meta if isinstance(meta, dict) else {}

    decision_surfaces: list[DecisionSurface] = []
    value_surfaces: list[DecisionSurface] = []
    for alt in alts:
        check_deadline()
        if not isinstance(alt, dict):
            continue
        kind = alt.get("kind")
        if kind not in {"DecisionSurface", "ValueDecisionSurface"}:
            continue
        inputs = alt.get("inputs")
        if not isinstance(inputs, list):
            continue
        site_path = None
        site_qual = None
        params: tuple[str, ...] = ()
        for entry in inputs:
            check_deadline()
            if not isinstance(entry, dict):
                continue
            entry_kind = entry.get("kind")
            entry_key = entry.get("key")
            if not isinstance(entry_kind, str) or not isinstance(entry_key, list):
                continue
            meta = node_meta.get((entry_kind, tuple(entry_key)), {})
            if entry_kind == "FunctionSite":
                site_path = meta.get("path") if isinstance(meta, dict) else None
                site_qual = meta.get("qual") if isinstance(meta, dict) else None
                if site_path is None and entry_key:
                    site_path = str(entry_key[0])
                if site_qual is None and len(entry_key) > 1:
                    site_qual = str(entry_key[1])
            elif entry_kind == "ParamSet":
                if isinstance(meta, dict) and isinstance(meta.get("params"), list):
                    params = tuple(str(p) for p in meta.get("params"))
                else:
                    params = tuple(str(p) for p in entry_key)

        if site_path is None or site_qual is None:
            continue
        evidence = alt.get("evidence")
        meta_text = ""
        if isinstance(evidence, dict):
            meta_text = str(evidence.get("meta") or "")
        surface = DecisionSurface(
            path=str(site_path),
            qual=str(site_qual),
            params=params,
            meta=meta_text,
        )
        if kind == "DecisionSurface":
            decision_surfaces.append(surface)
        else:
            value_surfaces.append(surface)

    return decision_surfaces, value_surfaces


def _parse_lint_entries(lines: Iterable[str]) -> list[LintEntry]:
    entries: list[LintEntry] = []
    for line in lines:
        check_deadline()
        parsed = _parse_lint_entry(line)
        if parsed is None:
            continue
        entries.append(parsed)
    return entries


def _render_bundle_candidates(
    bundle_counts: dict[tuple[str, ...], list[DecisionSurface]],
    *,
    max_examples: int,
) -> list[str]:
    lines: list[str] = []
    for params, surfaces in _sorted(
        bundle_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        check_deadline()
        if len(surfaces) < 2:
            continue
        bundle = ", ".join(params)
        lines.append(f"- Bundle candidate `{bundle}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:max_examples]:
            check_deadline()
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _render_param_clusters(
    param_to_surfaces: dict[str, list[DecisionSurface]],
    *,
    max_examples: int,
) -> list[str]:
    lines: list[str] = []
    for param, surfaces in _sorted(
        param_to_surfaces.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        check_deadline()
        if len(surfaces) < 2:
            continue
        lines.append(f"- Param `{param}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:max_examples]:
            check_deadline()
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _render_higher_order_candidates(
    bundle_counts: dict[tuple[str, ...], list[DecisionSurface]],
    *,
    min_functions: int,
    min_files: int,
    max_examples: int,
) -> list[str]:
    lines: list[str] = []
    for params, surfaces in _sorted(
        bundle_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        check_deadline()
        if len(surfaces) < min_functions:
            continue
        file_count = len({s.path for s in surfaces})
        if file_count < min_files:
            continue
        bundle = ", ".join(params)
        lines.append(
            f"- Higher-order bundle `{bundle}` appears in {len(surfaces)} functions "
            f"across {file_count} files:"
        )
        for surface in surfaces[:max_examples]:
            check_deadline()
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _build_suggestions(
    decision_surfaces: list[DecisionSurface],
    value_surfaces: list[DecisionSurface],
    config: ConsolidationConfig,
) -> dict[str, object]:
    bundle_counts: dict[tuple[str, ...], list[DecisionSurface]] = defaultdict(list)
    param_counts: dict[str, list[DecisionSurface]] = defaultdict(list)
    for surface in decision_surfaces:
        check_deadline()
        if surface.params:
            bundle_counts[tuple(_sorted(surface.params))].append(surface)
        for param in surface.params:
            check_deadline()
            param_counts[param].append(surface)

    bundle_suggestions: list[dict[str, object]] = []
    for params, surfaces in bundle_counts.items():
        check_deadline()
        if len(surfaces) < 2:
            continue
        boundary_count = sum(1 for s in surfaces if s.is_boundary)
        internal_count = len(surfaces) - boundary_count
        score = len(surfaces) + boundary_count * 2
        bundle_suggestions.append(
            {
                "params": list(params),
                "count": len(surfaces),
                "boundary_count": boundary_count,
                "internal_count": internal_count,
                "score": score,
                "sample_functions": [
                    f"{s.path}:{s.qual} ({s.meta})"
                    for s in surfaces[: config.max_examples]
                ],
            }
        )
    bundle_suggestions.sort(key=lambda item: (-item["score"], item["params"]))

    param_suggestions: list[dict[str, object]] = []
    for param, surfaces in param_counts.items():
        check_deadline()
        if len(surfaces) < 2:
            continue
        boundary_count = sum(1 for s in surfaces if s.is_boundary)
        internal_count = len(surfaces) - boundary_count
        score = len(surfaces) + boundary_count * 2
        param_suggestions.append(
            {
                "param": param,
                "count": len(surfaces),
                "boundary_count": boundary_count,
                "internal_count": internal_count,
                "score": score,
                "sample_functions": [
                    f"{s.path}:{s.qual} ({s.meta})"
                    for s in surfaces[: config.max_examples]
                ],
            }
        )
    param_suggestions.sort(key=lambda item: (-item["score"], item["param"]))

    higher_order: list[dict[str, object]] = []
    for params, surfaces in bundle_counts.items():
        check_deadline()
        if len(surfaces) < config.min_functions:
            continue
        file_count = len({s.path for s in surfaces})
        if file_count < config.min_files:
            continue
        boundary_count = sum(1 for s in surfaces if s.is_boundary)
        internal_count = len(surfaces) - boundary_count
        score = len(surfaces) + file_count * 2 + boundary_count
        higher_order.append(
            {
                "params": list(params),
                "count": len(surfaces),
                "file_count": file_count,
                "boundary_count": boundary_count,
                "internal_count": internal_count,
                "score": score,
                "sample_functions": [
                    f"{s.path}:{s.qual} ({s.meta})"
                    for s in surfaces[: config.max_examples]
                ],
            }
        )
    higher_order.sort(key=lambda item: (-item["score"], item["params"]))

    value_decision = [
        {
            "params": list(surface.params),
            "qual": surface.qual,
            "path": surface.path,
            "meta": surface.meta,
        }
        for surface in value_surfaces
    ]

    return {
        "bundle_candidates": bundle_suggestions,
        "higher_order_bundles": higher_order,
        "param_clusters": param_suggestions,
        "value_decision_surfaces": value_decision,
    }




def _write_consolidation_report(
    output_path: Path,
    decision_surfaces: list[DecisionSurface],
    value_surfaces: list[DecisionSurface],
    lint_entries: list[LintEntry],
    config: ConsolidationConfig,
    source_mode: str,
    fallback_notes: list[str] | None = None,
) -> None:
    boundary_surfaces = [s for s in decision_surfaces if s.is_boundary]
    param_to_surfaces: dict[str, list[DecisionSurface]] = defaultdict(list)
    bundle_counts: dict[tuple[str, ...], list[DecisionSurface]] = defaultdict(list)
    for surface in decision_surfaces:
        check_deadline()
        if surface.params:
            bundle_counts[tuple(_sorted(surface.params))].append(surface)
        for param in surface.params:
            check_deadline()
            param_to_surfaces[param].append(surface)

    lint_by_code = Counter(entry.code for entry in lint_entries)
    lint_by_file = Counter(entry.path for entry in lint_entries)

    lines: list[str] = []
    lines.append("# Consolidation audit (decision surfaces)")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        f"- Decision surfaces: {len(decision_surfaces)} (boundary: {len(boundary_surfaces)})"
    )
    lines.append(f"- Value-encoded decision surfaces: {len(value_surfaces)}")
    lines.append(f"- Lint findings: {len(lint_entries)}")
    lines.append(
        f"- Higher-order thresholds: min_functions={config.min_functions}, "
        f"min_files={config.min_files}, max_examples={config.max_examples}"
    )
    lines.append(f"- Forest required: {config.require_forest}")
    lines.append(f"- Consolidation source mode: {source_mode}")
    if fallback_notes:
        lines.append(
            f"- {FOREST_FALLBACK_MARKER}: " + "; ".join(_sorted(set(fallback_notes)))
        )
    if lint_by_code:
        lines.append("- Lint codes: " + ", ".join(
            f"{code}={count}" for code, count in lint_by_code.most_common()
        ))
    lines.append("")

    lines.append("## Bundle candidates (repeated param sets)")
    bundle_lines = _render_bundle_candidates(bundle_counts, max_examples=config.max_examples)
    lines.extend(bundle_lines if bundle_lines else ["- None (no repeated param sets)."])
    lines.append("")

    lines.append("## Higher-order bundles (repeated param sets across files)")
    higher_order_lines = _render_higher_order_candidates(
        bundle_counts,
        min_functions=config.min_functions,
        min_files=config.min_files,
        max_examples=config.max_examples,
    )
    lines.extend(
        higher_order_lines
        if higher_order_lines
        else [
            "- None (no repeated param sets across files at configured thresholds)."
        ]
    )
    lines.append("")

    lines.append("## Param clusters (repeated params)")
    cluster_lines = _render_param_clusters(param_to_surfaces, max_examples=config.max_examples)
    lines.extend(cluster_lines if cluster_lines else ["- None (no repeated params)."])
    lines.append("")

    lines.append("## Boundary decision lint locations (top 20)")
    boundary_lint = [
        entry
        for entry in lint_entries
        if entry.code in {"GABION_DECISION_SURFACE", "GABION_VALUE_DECISION_SURFACE"}
    ]
    boundary_lint_sorted = _sorted(boundary_lint, key=lambda e: (e.path, e.line, e.col))
    if boundary_lint_sorted:
        for entry in boundary_lint_sorted[:20]:
            check_deadline()
            lines.append(
                f"- {entry.path}:{entry.line}:{entry.col} {entry.code} {entry.message}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Value-encoded decision surfaces")
    if value_surfaces:
        for surface in value_surfaces[:20]:
            check_deadline()
            params = ", ".join(surface.params)
            lines.append(f"- {surface.path}:{surface.qual} ({params}; {surface.meta})")
    else:
        lines.append("- None")
    lines.append("")

    if lint_by_file:
        lines.append("## Top lint files")
        for path, count in lint_by_file.most_common(10):
            check_deadline()
            lines.append(f"- {path}: {count}")
        lines.append("")

    output_path.write_text("\n".join(lines))


# --- Lint summary helpers ---


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line for line in path.read_text().splitlines() if line.strip()]


def _summarize_lint(lines: Iterable[str]) -> dict[str, object]:
    codes = Counter()
    files = Counter()
    total = 0
    by_code_file: dict[str, Counter[str]] = defaultdict(Counter)
    for line in lines:
        check_deadline()
        parsed = _parse_lint_entry(line)
        if parsed is None:
            continue
        total += 1
        codes[parsed.code] += 1
        files[parsed.path] += 1
        by_code_file[parsed.code][parsed.path] += 1
    return {
        "total": total,
        "codes": dict(codes.most_common()),
        "files": dict(files.most_common()),
        "by_code_file": {
            code: dict(counter.most_common()) for code, counter in by_code_file.items()
        },
    }


# --- CLI command handlers ---


def _docflow_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        sppf_gh_ref_mode = _resolve_sppf_gh_ref_mode(args.sppf_gh_ref_mode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    docflow_result, docs = run_docflow_domain(
        root=root,
        extra_paths=args.extra_path,
        extra_strict=args.extra_strict,
        sppf_gh_ref_mode=sppf_gh_ref_mode,
        baseline_write_emitted=bool(args.baseline_write_emitted),
        delta_guard_checked=bool(args.delta_guard_checked),
        build_context=_docflow_audit_context,
        load_docs=_load_docflow_docs,
        evaluate_obligations=_evaluate_docflow_obligations,
    )
    context = docflow_result.context
    obligations = docflow_result.obligations
    warnings = docflow_result.warnings
    violations = docflow_result.violations
    _aggregate = GovernanceAuditAggregateResult(docflow=docflow_result)
    _emit_docflow_suite_artifacts(
        root=root,
        extra_paths=args.extra_path,
        issues_json=args.issues_json,
        forest_output=args.suite_forest,
        relation_output=args.suite_relation,
    )
    _emit_docflow_compliance(
        rows=context.invariant_rows,
        invariants=context.invariants,
        json_output=args.compliance_json,
        md_output=args.compliance_md,
        obligations=obligations,
    )
    _emit_docflow_canonicality(
        root=root,
        extra_paths=args.extra_path,
        json_output=args.canonicality_json,
        md_output=args.canonicality_md,
    )
    _emit_docflow_cycles(
        docs,
        json_output=args.cycles_json,
        md_output=args.cycles_md,
    )
    _emit_docflow_change_protocol(
        docs,
        json_output=args.change_protocol_json,
        md_output=args.change_protocol_md,
    )
    _emit_docflow_section_reviews(
        docs,
        context.revisions,
        json_output=args.section_reviews_json,
        md_output=args.section_reviews_md,
    )
    _emit_docflow_implication_matrices(
        docs=docs,
        json_output=args.implication_matrix_json,
    )
    graph_warnings, graph_violations = _agent_instruction_graph(
        root=root,
        docs=docs,
        json_output=args.agent_instruction_graph_json,
        md_output=args.agent_instruction_graph_md,
    )
    warnings = warnings + graph_warnings
    violations = violations + graph_violations

    print("Docflow audit summary")
    if warnings:
        print("Warnings:")
        for w in warnings:
            check_deadline()
            print(f"- {w}")
    if violations:
        print("Violations:")
        for v in violations:
            check_deadline()
            print(f"- {v}")
    if not warnings and not violations:
        print("No issues detected.")

    if (warnings or violations) and args.fail_on_violations:
        return 1
    return 0


def _sppf_graph_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    graph_result = build_sppf_graph(
        root=root,
        issues_json=args.issues_json,
        build_graph=_build_sppf_dependency_graph,
    )
    _aggregate = GovernanceAuditAggregateResult(sppf_graph=graph_result)
    _write_sppf_graph_outputs(
        graph_result.graph,
        json_output=args.json_output,
        dot_output=args.dot_output,
    )
    print(f"SPPF dependency graph written to {args.json_output}")
    if args.dot_output is not None:
        print(f"SPPF dependency graph DOT written to {args.dot_output}")
    return 0


def _status_consistency_command(
    args: argparse.Namespace,
    *,
    run_status_consistency_fn: Callable[..., SppfStatusConsistencyResult] = run_status_consistency,
) -> int:
    root = Path(args.root)
    status_result = run_status_consistency_fn(
        root=root,
        extra_paths=args.extra_path,
        load_docs=_load_docflow_docs,
        axis_audit=_sppf_axis_audit,
        sync_check=_sppf_sync_check,
    )
    violations = status_result.violations
    warnings = status_result.warnings
    rendered = render_compliance(status_result)
    _aggregate = GovernanceAuditAggregateResult(
        sppf_status=status_result,
        compliance_render=rendered,
    )
    payload = {
        "root": str(root),
        **status_result.payload,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"SPPF status consistency JSON written to {args.json_output}")
    if args.md_output is not None:
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(rendered.status_consistency.markdown, encoding="utf-8")
        print(f"SPPF status consistency markdown written to {args.md_output}")
    if violations and args.fail_on_violations:
        return 1
    return 0


def _decision_tiers_command(args: argparse.Namespace) -> int:
    lint_path = args.lint or _latest_lint_path(args.root)
    return _decision_tier_candidates(lint_path, tier=args.tier, output_format=args.format)


def _consolidation_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    decision_path = Path(args.decision) if args.decision is not None else None
    lint_path = Path(args.lint) if args.lint is not None else None
    output_path = Path(args.output) if args.output is not None else None
    snapshot_dir = None
    if decision_path is None or lint_path is None or output_path is None:
        snapshot_dir = _latest_snapshot_dir(root)
    if decision_path is None:
        decision_path = snapshot_dir / "decision_snapshot.json"
    if lint_path is None:
        lint_path = snapshot_dir / "lint.txt"
    if output_path is None:
        output_path = snapshot_dir / "consolidation_report.md"

    config = _load_consolidation_config(root)
    decision_obj = json.loads(decision_path.read_text())
    decision_lines = decision_obj.get("decision_surfaces", [])
    value_lines = decision_obj.get("value_decision_surfaces", [])
    forest_obj = decision_obj.get("forest")

    decision_surfaces: list[DecisionSurface]
    value_surfaces: list[DecisionSurface]
    fallback_notes: list[str] = []
    forest_used = False
    if isinstance(forest_obj, dict):
        decision_surfaces, value_surfaces = _surfaces_from_forest(forest_obj)
        if decision_surfaces or value_surfaces or (not decision_lines and not value_lines):
            forest_used = True
        else:
            fallback_notes.append("forest missing decision/value surface alts")
    else:
        fallback_notes.append("missing forest payload")

    if not forest_used:
        if config.require_forest and not args.allow_fallback:
            raise SystemExit(
                "forest-only mode enabled but decision snapshot forest is missing/incomplete; "
                "rerun gabion audit with --emit-decision-snapshot or set require_forest=false "
                "or pass --allow-fallback"
            )
        decision_surfaces = _parse_surfaces(decision_lines, value_encoded=False)
        value_surfaces = _parse_surfaces(value_lines, value_encoded=True)
    source_mode = (
        CONSOLIDATION_SOURCE_FOREST_NATIVE if forest_used else CONSOLIDATION_SOURCE_FALLBACK_DERIVED
    )
    lint_entries = _parse_lint_entries(lint_path.read_text().splitlines())

    _write_consolidation_report(
        output_path,
        decision_surfaces,
        value_surfaces,
        lint_entries,
        config,
        source_mode=source_mode,
        fallback_notes=fallback_notes if not forest_used else None,
    )
    if not forest_used:
        print(
            json.dumps(
                {
                    "warning_class": FOREST_FALLBACK_WARNING_CLASS,
                    "source_mode": source_mode,
                    "notes": _sorted(set(fallback_notes)),
                },
                sort_keys=True,
            )
        )
    if args.json_output is not None:
        suggestions = _build_suggestions(decision_surfaces, value_surfaces, config)
        suggestions["report_metadata"] = {
            "source_mode": source_mode,
            "forest_required": config.require_forest,
            "fallback_used": not forest_used,
        }
        args.json_output.write_text(json.dumps(suggestions, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


def _lint_summary_command(args: argparse.Namespace) -> int:
    lint_path = args.lint or _latest_lint_path(args.root)
    lines = _load_lines(lint_path)
    summary = _summarize_lint(lines)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    total = summary["total"]
    print(f"Lint summary for {lint_path} ({total} findings)")
    print("\nTop codes:")
    for code, count in list(summary["codes"].items())[: args.top]:
        check_deadline()
        print(f"- {code}: {count}")
    print("\nTop files:")
    for path, count in list(summary["files"].items())[: args.top]:
        check_deadline()
        print(f"- {path}: {count}")
    return 0


def _add_docflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Additional doc path(s) or directories to include (repeatable).",
    )
    parser.add_argument(
        "--extra-strict",
        action="store_true",
        help="Audit extra paths as full docs (require frontmatter + full checks).",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if warnings or violations are detected",
    )
    parser.add_argument(
        "--sppf-gh-ref-mode",
        default="required",
        choices=("required", "advisory"),
        help="SPPF GH-reference enforcement mode (default: required).",
    )
    parser.add_argument(
        "--baseline-write-emitted",
        action="store_true",
        help="Declare that this run emits a baseline write (for obligation checks).",
    )
    parser.add_argument(
        "--delta-guard-checked",
        action="store_true",
        help="Declare that delta guard checks were executed before baseline write.",
    )
    parser.add_argument(
        "--issues-json",
        type=Path,
        default=None,
        help="Optional GH issues JSON payload (gh issue list --json ...) for suite metadata.",
    )
    parser.add_argument(
        "--suite-forest",
        type=Path,
        default=Path("artifacts/docflow_suite_forest.json"),
        help="Output path for docflow SuiteSite forest JSON.",
    )
    parser.add_argument(
        "--suite-relation",
        type=Path,
        default=Path("artifacts/docflow_suite_relation.json"),
        help="Output path for docflow SuiteSite relation JSON.",
    )
    parser.add_argument(
        "--compliance-json",
        type=Path,
        default=Path("artifacts/out/docflow_compliance.json"),
        help="Output path for docflow compliance JSON.",
    )
    parser.add_argument(
        "--compliance-md",
        type=Path,
        default=Path("artifacts/audit_reports/docflow_compliance.md"),
        help="Output path for docflow compliance markdown.",
    )
    parser.add_argument(
        "--canonicality-json",
        type=Path,
        default=Path("artifacts/out/docflow_canonicality.json"),
        help="Output path for docflow canonicality JSON.",
    )
    parser.add_argument(
        "--canonicality-md",
        type=Path,
        default=Path("artifacts/audit_reports/docflow_canonicality.md"),
        help="Output path for docflow canonicality markdown.",
    )
    parser.add_argument(
        "--cycles-json",
        type=Path,
        default=Path("artifacts/out/docflow_cycles.json"),
        help="Output path for docflow dependency cycle JSON.",
    )
    parser.add_argument(
        "--cycles-md",
        type=Path,
        default=Path("artifacts/audit_reports/docflow_cycles.md"),
        help="Output path for docflow dependency cycle markdown.",
    )
    parser.add_argument(
        "--change-protocol-json",
        type=Path,
        default=Path("artifacts/out/docflow_change_protocol.json"),
        help="Output path for docflow change-protocol JSON.",
    )
    parser.add_argument(
        "--change-protocol-md",
        type=Path,
        default=Path("artifacts/audit_reports/docflow_change_protocol.md"),
        help="Output path for docflow change-protocol markdown.",
    )
    parser.add_argument(
        "--section-reviews-json",
        type=Path,
        default=Path("artifacts/out/docflow_section_reviews.json"),
        help="Output path for docflow anchor review JSON.",
    )
    parser.add_argument(
        "--section-reviews-md",
        type=Path,
        default=Path("artifacts/audit_reports/docflow_section_reviews.md"),
        help="Output path for docflow anchor review markdown.",
    )
    parser.add_argument(
        "--implication-matrix-json",
        type=Path,
        default=Path("artifacts/out/docflow_implication_matrices.json"),
        help="Output path for docflow implication lattice and dependency-matrix JSON.",
    )
    parser.add_argument(
        "--agent-instruction-graph-json",
        type=Path,
        default=Path("artifacts/out/agent_instruction_drift.json"),
        help="Output path for the agent instruction graph drift JSON.",
    )
    parser.add_argument(
        "--agent-instruction-graph-md",
        type=Path,
        default=Path("artifacts/audit_reports/agent_instruction_drift.md"),
        help="Output path for the agent instruction graph drift markdown.",
    )
    parser.set_defaults(func=_docflow_command)


def _add_decision_tier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lint", type=Path, default=None, help="Path to lint.txt")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--tier",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="Tier to emit candidates for (default: 3).",
    )
    parser.add_argument(
        "--format",
        choices=("toml", "lines"),
        default="toml",
        help="Output format (default: toml).",
    )
    parser.set_defaults(func=_decision_tiers_command)


def _add_consolidation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument("--decision", type=Path, default=None, help="decision_snapshot.json")
    parser.add_argument("--lint", type=Path, default=None, help="lint.txt")
    parser.add_argument("--output", type=Path, default=None, help="Output report path")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path for consolidation suggestions.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow explicit fallback parsing when forest data is missing/incomplete.",
    )
    parser.set_defaults(func=_consolidation_command)


def _add_lint_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lint", type=Path, default=None, help="Path to lint.txt")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--top", type=int, default=10, help="Show top N entries")
    parser.set_defaults(func=_lint_summary_command)


def _add_sppf_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/sppf_dependency_graph.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--dot-output",
        type=Path,
        default=None,
        help="Optional Graphviz DOT output path",
    )
    parser.add_argument(
        "--issues-json",
        type=Path,
        default=None,
        help="Optional GH issues JSON payload (gh issue list --json ...) for titles/labels.",
    )
    parser.set_defaults(func=_sppf_graph_command)


def _add_status_consistency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument(
        "--extra-path",
        action="append",
        default=None,
        help="Additional markdown path to include in docflow parsing.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("artifacts/out/status_consistency.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("artifacts/audit_reports/status_consistency.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero when violations are detected.",
    )
    parser.set_defaults(func=_status_consistency_command)


def _parse_single_command_args(
    add_args: Callable[[argparse.ArgumentParser], None], argv: list[str] | None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args(_coerce_argv(argv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit tooling bundle (docflow, consolidation, lint summary)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_docflow_args(subparsers.add_parser("docflow", help="Run docflow audit."))
    _add_decision_tier_args(
        subparsers.add_parser(
            "decision-tiers", help="Extract decision-tier candidates from lint."
        )
    )
    _add_consolidation_args(
        subparsers.add_parser(
            "consolidation", help="Generate consolidation audit report."
        )
    )
    _add_lint_summary_args(
        subparsers.add_parser("lint-summary", help="Summarize lint output.")
    )
    _add_sppf_graph_args(
        subparsers.add_parser("sppf-graph", help="Emit SPPF dependency graph.")
    )
    _add_status_consistency_args(
        subparsers.add_parser(
            "status-consistency", help="Run SPPF checklist/influence consistency checks."
        )
    )

    with _audit_deadline_scope():
        args = parser.parse_args(argv)
        return int(args.func(args))


def run_docflow_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_docflow_args, argv)
        return _docflow_command(args)


def run_decision_tiers_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_decision_tier_args, argv)
        return _decision_tiers_command(args)


def run_consolidation_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_consolidation_args, argv)
        return _consolidation_command(args)


def run_lint_summary_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_lint_summary_args, argv)
        return _lint_summary_command(args)


def run_sppf_graph_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_sppf_graph_args, argv)
        return _sppf_graph_command(args)


def run_status_consistency_cli(argv: list[str] | None = None) -> int:
    with _audit_deadline_scope():
        args = _parse_single_command_args(_add_status_consistency_args, argv)
        return _status_consistency_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
