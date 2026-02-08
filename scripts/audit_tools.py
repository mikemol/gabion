#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, TypeAlias


# --- Docflow audit constants ---

CORE_GOVERNANCE_DOCS = [
    "POLICY_SEED.md",
    "glossary.md",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
]

GOVERNANCE_DOCS = CORE_GOVERNANCE_DOCS + [
    "docs/publishing_practices.md",
    "docs/influence_index.md",
    "docs/coverage_semantics.md",
    "docs/matrix_acceptance.md",
]

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
}

FrontmatterScalar: TypeAlias = str | int
FrontmatterValue: TypeAlias = FrontmatterScalar | List[str] | dict[str, FrontmatterScalar]
Frontmatter: TypeAlias = dict[str, FrontmatterValue]

# --- Lint parsing helpers ---

PARAM_RE = re.compile(r"param '([^']+)'\s*\(")

# --- Consolidation regex helpers ---

DECISION_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<qual>\S+) decision surface params: (?P<params>.+) \((?P<meta>[^)]+)\)$"
)
VALUE_DECISION_RE = re.compile(
    r"^(?P<path>[^:]+):(?P<qual>\S+) value-encoded decision params: (?P<params>.+) \((?P<meta>[^)]+)\)$"
)
FOREST_FALLBACK_MARKER = "FOREST_FALLBACK_USED"


@dataclass(frozen=True)
class Doc:
    frontmatter: Frontmatter
    body: str


@dataclass(frozen=True)
class DecisionSurface:
    path: str
    qual: str
    params: tuple[str, ...]
    meta: str

    @property
    def is_boundary(self) -> bool:
        return "boundary" in self.meta


@dataclass(frozen=True)
class LintEntry:
    path: str
    line: int
    col: int
    code: str
    message: str
    param: str | None


@dataclass(frozen=True)
class ConsolidationConfig:
    min_functions: int = 3
    min_files: int = 2
    max_examples: int = 5
    require_forest: bool = False


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
        require_forest=_coerce_bool("require_forest", False),
    )


def _parse_lint_entry(line: str) -> LintEntry | None:
    match = re.match(r"^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+):\s*(?P<rest>.*)$", line.strip())
    if not match:
        return None
    path = match.group("path")
    try:
        line_no = int(match.group("line"))
        col_no = int(match.group("col"))
    except ValueError:
        return None
    remainder = match.group("rest")
    remainder_parts = remainder.split(" ", 1) if remainder else []
    if not remainder_parts:
        return None
    code = remainder_parts[0]
    message = remainder_parts[1] if len(remainder_parts) > 1 else ""
    param = None
    param_match = PARAM_RE.search(message)
    if param_match:
        param = param_match.group(1)
    return LintEntry(
        path=path,
        line=line_no,
        col=col_no,
        code=code,
        message=message,
        param=param,
    )


# --- Docflow audit helpers ---


def _parse_frontmatter(text: str) -> tuple[Frontmatter, str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    fm_lines: List[str] = []
    idx = 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            break
        fm_lines.append(line)
        idx += 1
    body = "\n".join(lines[idx:])
    return _parse_yaml_like(fm_lines), body


def _parse_yaml_like(lines: List[str]) -> Frontmatter:
    data: Frontmatter = {}
    current_list_key: str | None = None
    current_map_key: str | None = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        if current_map_key is not None and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            parsed: FrontmatterScalar = int(value) if value.isdigit() else value
            mapping = data.get(current_map_key)
            if not isinstance(mapping, dict):
                mapping = {}
            mapping[key] = parsed
            data[current_map_key] = mapping
            continue
        if line.lstrip().startswith("- "):
            if current_list_key is None:
                continue
            item = line.strip()[2:].strip()
            if isinstance(data.get(current_list_key), list):
                data[current_list_key].append(item)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                if key in MAP_FIELDS:
                    data[key] = {}
                    current_map_key = key
                    current_list_key = None
                else:
                    data[key] = []
                    current_list_key = key
                    current_map_key = None
                continue
            if value.startswith("\"") and value.endswith("\""):
                value = value[1:-1]
            if value.isdigit():
                data[key] = int(value)
            else:
                data[key] = value
            current_list_key = None
            current_map_key = None
    return data


def _lower_name(path: Path) -> str:
    return path.stem.lower()


def _docflow_audit(root: Path) -> Tuple[List[str], List[str]]:
    violations: List[str] = []
    warnings: List[str] = []

    docs: dict[str, Doc] = {}
    doc_ids: dict[str, str] = {}

    for rel in GOVERNANCE_DOCS:
        path = root / rel
        if not path.exists():
            violations.append(f"missing governance doc: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        docs[rel] = Doc(frontmatter=fm, body=body)
        doc_id = fm.get("doc_id")
        if isinstance(doc_id, str):
            if doc_id in doc_ids:
                violations.append(
                    f"duplicate doc_id '{doc_id}' in {rel} and {doc_ids[doc_id]}"
                )
            else:
                doc_ids[doc_id] = rel

    governance_set = set(GOVERNANCE_DOCS)
    core_set = set(CORE_GOVERNANCE_DOCS)

    revisions: dict[str, int] = {}
    for rel, payload in docs.items():
        fm = payload.frontmatter
        if isinstance(fm.get("doc_revision"), int):
            revisions[rel] = fm["doc_revision"]

    for rel, payload in docs.items():
        fm = payload.frontmatter
        body = payload.body
        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in fm:
                violations.append(f"{rel}: missing frontmatter field '{field}'")
        # Required list for doc_scope/doc_requires
        for field in ("doc_scope", "doc_requires"):
            if field in fm and not isinstance(fm[field], list):
                violations.append(f"{rel}: frontmatter field '{field}' must be a list")
        for field in ("doc_reviewed_as_of", "doc_review_notes"):
            if field in fm and not isinstance(fm[field], dict):
                violations.append(
                    f"{rel}: frontmatter field '{field}' must be a map"
                )
        # Normative docs must require the governance bundle.
        if fm.get("doc_authority") == "normative":
            requires = set(fm.get("doc_requires", []))
            expected = core_set - {rel}
            missing = sorted(expected - requires)
            if missing:
                violations.append(
                    f"{rel}: missing required governance references: {', '.join(missing)}"
                )
        # Commutation symmetry
        commutes = fm.get("doc_commutes_with")
        if isinstance(commutes, list):
            for other in commutes:
                other_doc = docs.get(other)
                if other_doc is None:
                    violations.append(f"{rel}: doc_commutes_with target missing: {other}")
                    continue
                other_fm = other_doc.frontmatter
                other_commutes = other_fm.get("doc_commutes_with", [])
                if not isinstance(other_commutes, list) or rel not in other_commutes:
                    violations.append(
                        f"{rel}: commutation with {other} not reciprocated"
                    )
        # Body references vs frontmatter requirements
        requires = fm.get("doc_requires", [])
        if isinstance(requires, list):
            body_lower = body.lower()
            for req in requires:
                if req in body:
                    continue
                req_name = _lower_name(Path(req))
                if req_name and req_name in body_lower:
                    warnings.append(
                        f"{rel}: implicit reference to {req} (Tier-2); prefer explicit path"
                    )
                else:
                    violations.append(f"{rel}: missing explicit reference to {req}")
        # Convergence check (re-reviewed as of)
        reviewed = fm.get("doc_reviewed_as_of")
        review_notes = fm.get("doc_review_notes")
        if isinstance(requires, list) and requires:
            if not isinstance(reviewed, dict):
                violations.append(f"{rel}: doc_reviewed_as_of missing or invalid")
            else:
                for req in requires:
                    expected = revisions.get(req)
                    if expected is None:
                        violations.append(
                            f"{rel}: doc_reviewed_as_of cannot resolve {req}"
                        )
                        continue
                    seen = reviewed.get(req)
                    if not isinstance(seen, int):
                        violations.append(
                            f"{rel}: doc_reviewed_as_of[{req}] must be an integer"
                        )
                    elif seen != expected:
                        violations.append(
                            f"{rel}: doc_reviewed_as_of[{req}]={seen} does not match {expected}"
                        )
            if not isinstance(review_notes, dict):
                violations.append(f"{rel}: doc_review_notes missing or invalid")
            else:
                for req in requires:
                    note = review_notes.get(req)
                    if not isinstance(note, str) or not note.strip():
                        violations.append(
                            f"{rel}: doc_review_notes[{req}] missing or empty"
                        )

    warnings.extend(_tooling_warnings(root, docs))
    warnings.extend(_influence_warnings(root))

    _ = governance_set
    return violations, warnings


def _tooling_warnings(root: Path, docs: dict[str, Doc]) -> List[str]:
    warnings: List[str] = []
    makefile = root / "Makefile"
    if makefile.exists():
        for rel in ("README.md", "CONTRIBUTING.md"):
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
        if "scripts/checks.sh" not in body:
            warnings.append(
                "CONTRIBUTING.md: scripts/checks.sh present but not documented"
            )
    return warnings


def _influence_warnings(root: Path) -> List[str]:
    warnings: List[str] = []
    inbox = root / "in"
    index_path = root / "docs" / "influence_index.md"
    if not inbox.exists():
        return warnings
    if not index_path.exists():
        warnings.append("docs/influence_index.md: missing influence index for in/")
        return warnings
    index_text = index_path.read_text(encoding="utf-8")
    for path in sorted(inbox.glob("in-*.md")):
        rel = path.as_posix()
        if rel not in index_text:
            warnings.append(f"docs/influence_index.md: missing {rel}")
    return warnings


# --- Decision tier candidate helpers ---


def _decision_tier_candidates(lint_path: Path, *, tier: int, output_format: str) -> int:
    codes = {"GABION_DECISION_SURFACE", "GABION_VALUE_DECISION_SURFACE"}
    keys: list[str] = []
    for line in lint_path.read_text().splitlines():
        parsed = _parse_lint_entry(line)
        if parsed is None:
            continue
        if parsed.code not in codes:
            continue
        keys.append(f"{parsed.path}:{parsed.line}:{parsed.col}")

    keys = sorted(set(keys))
    if output_format == "lines":
        for key in keys:
            print(key)
        return 0

    tier_key = f"tier{tier}"
    print("[decision]")
    print(f"{tier_key} = [")
    for key in keys:
        print(f"  \"{key}\",")
    print("]")
    return 0


# --- Consolidation audit helpers ---


def _parse_surfaces(lines: Iterable[str], *, value_encoded: bool) -> list[DecisionSurface]:
    surfaces: list[DecisionSurface] = []
    pattern = VALUE_DECISION_RE if value_encoded else DECISION_RE
    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        params = [p.strip() for p in match.group("params").split(",") if p.strip()]
        surfaces.append(
            DecisionSurface(
                path=match.group("path"),
                qual=match.group("qual"),
                params=tuple(params),
                meta=match.group("meta"),
            )
        )
    return surfaces


def _surfaces_from_forest(forest: dict[str, object]) -> tuple[list[DecisionSurface], list[DecisionSurface]]:
    nodes = forest.get("nodes")
    alts = forest.get("alts")
    if not isinstance(nodes, list) or not isinstance(alts, list):
        return [], []

    node_meta: dict[tuple[str, tuple[object, ...]], dict[str, object]] = {}
    for node in nodes:
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
    for params, surfaces in sorted(
        bundle_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        if len(surfaces) < 2:
            continue
        bundle = ", ".join(params)
        lines.append(f"- Bundle candidate `{bundle}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:max_examples]:
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _render_param_clusters(
    param_to_surfaces: dict[str, list[DecisionSurface]],
    *,
    max_examples: int,
) -> list[str]:
    lines: list[str] = []
    for param, surfaces in sorted(
        param_to_surfaces.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        if len(surfaces) < 2:
            continue
        lines.append(f"- Param `{param}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:max_examples]:
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
    for params, surfaces in sorted(
        bundle_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
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
        if surface.params:
            bundle_counts[tuple(sorted(surface.params))].append(surface)
        for param in surface.params:
            param_counts[param].append(surface)

    bundle_suggestions: list[dict[str, object]] = []
    for params, surfaces in bundle_counts.items():
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
    fallback_notes: list[str] | None = None,
) -> None:
    boundary_surfaces = [s for s in decision_surfaces if s.is_boundary]
    param_to_surfaces: dict[str, list[DecisionSurface]] = defaultdict(list)
    bundle_counts: dict[tuple[str, ...], list[DecisionSurface]] = defaultdict(list)
    for surface in decision_surfaces:
        if surface.params:
            bundle_counts[tuple(sorted(surface.params))].append(surface)
        for param in surface.params:
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
    if fallback_notes:
        lines.append(
            f"- {FOREST_FALLBACK_MARKER}: " + "; ".join(sorted(set(fallback_notes)))
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
    boundary_lint_sorted = sorted(boundary_lint, key=lambda e: (e.path, e.line, e.col))
    if boundary_lint_sorted:
        for entry in boundary_lint_sorted[:20]:
            lines.append(
                f"- {entry.path}:{entry.line}:{entry.col} {entry.code} {entry.message}"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Value-encoded decision surfaces")
    if value_surfaces:
        for surface in value_surfaces[:20]:
            params = ", ".join(surface.params)
            lines.append(f"- {surface.path}:{surface.qual} ({params}; {surface.meta})")
    else:
        lines.append("- None")
    lines.append("")

    if lint_by_file:
        lines.append("## Top lint files")
        for path, count in lint_by_file.most_common(10):
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
    violations, warnings = _docflow_audit(root)

    print("Docflow audit summary")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"- {v}")
    if not warnings and not violations:
        print("No issues detected.")

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
        if config.require_forest:
            raise SystemExit(
                "forest-only mode enabled but decision snapshot forest is missing/incomplete; "
                "rerun gabion audit with --emit-decision-snapshot or set require_forest=false"
            )
        decision_surfaces = _parse_surfaces(decision_lines, value_encoded=False)
        value_surfaces = _parse_surfaces(value_lines, value_encoded=True)
    lint_entries = _parse_lint_entries(lint_path.read_text().splitlines())

    _write_consolidation_report(
        output_path,
        decision_surfaces,
        value_surfaces,
        lint_entries,
        config,
        fallback_notes=fallback_notes if not forest_used else None,
    )
    if args.json_output is not None:
        suggestions = _build_suggestions(decision_surfaces, value_surfaces, config)
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
        print(f"- {code}: {count}")
    print("\nTop files:")
    for path, count in list(summary["files"].items())[: args.top]:
        print(f"- {path}: {count}")
    return 0


def _add_docflow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if violations are detected",
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
    parser.set_defaults(func=_consolidation_command)


def _add_lint_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lint", type=Path, default=None, help="Path to lint.txt")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repo root")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    parser.add_argument("--top", type=int, default=10, help="Show top N entries")
    parser.set_defaults(func=_lint_summary_command)


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

    args = parser.parse_args(argv)
    return int(args.func(args))


def run_docflow_cli(argv: list[str] | None = None) -> int:
    args = _parse_single_command_args(_add_docflow_args, argv)
    return _docflow_command(args)


def run_decision_tiers_cli(argv: list[str] | None = None) -> int:
    args = _parse_single_command_args(_add_decision_tier_args, argv)
    return _decision_tiers_command(args)


def run_consolidation_cli(argv: list[str] | None = None) -> int:
    args = _parse_single_command_args(_add_consolidation_args, argv)
    return _consolidation_command(args)


def run_lint_summary_cli(argv: list[str] | None = None) -> int:
    args = _parse_single_command_args(_add_lint_summary_args, argv)
    return _lint_summary_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
