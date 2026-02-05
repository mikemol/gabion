#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, TypeAlias


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


def _parse_lint_entry(line: str) -> LintEntry | None:
    parts = line.strip().split(": ", 1)
    if len(parts) != 2:
        return None
    location, remainder = parts
    loc_parts = location.split(":")
    if len(loc_parts) < 3:
        return None
    path = ":".join(loc_parts[:-2])
    try:
        line_no = int(loc_parts[-2])
        col_no = int(loc_parts[-1])
    except ValueError:
        return None
    remainder_parts = remainder.split(" ", 1)
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


def _parse_lint_entries(lines: Iterable[str]) -> list[LintEntry]:
    entries: list[LintEntry] = []
    for line in lines:
        parsed = _parse_lint_entry(line)
        if parsed is None:
            continue
        entries.append(parsed)
    return entries


def _render_bundle_candidates(bundle_counts: dict[tuple[str, ...], list[DecisionSurface]]) -> list[str]:
    lines: list[str] = []
    for params, surfaces in sorted(
        bundle_counts.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        if len(surfaces) < 2:
            continue
        bundle = ", ".join(params)
        lines.append(f"- Bundle candidate `{bundle}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:5]:
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _render_param_clusters(param_to_surfaces: dict[str, list[DecisionSurface]]) -> list[str]:
    lines: list[str] = []
    for param, surfaces in sorted(
        param_to_surfaces.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        if len(surfaces) < 2:
            continue
        lines.append(f"- Param `{param}` appears in {len(surfaces)} functions:")
        for surface in surfaces[:5]:
            lines.append(f"  - {surface.path}:{surface.qual} ({surface.meta})")
    return lines


def _build_suggestions(
    decision_surfaces: list[DecisionSurface],
    value_surfaces: list[DecisionSurface],
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
                    f"{s.path}:{s.qual} ({s.meta})" for s in surfaces[:5]
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
                    f"{s.path}:{s.qual} ({s.meta})" for s in surfaces[:5]
                ],
            }
        )
    param_suggestions.sort(key=lambda item: (-item["score"], item["param"]))

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
        "param_clusters": param_suggestions,
        "value_decision_surfaces": value_decision,
    }




def _write_consolidation_report(
    output_path: Path,
    decision_surfaces: list[DecisionSurface],
    value_surfaces: list[DecisionSurface],
    lint_entries: list[LintEntry],
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
    if lint_by_code:
        lines.append("- Lint codes: " + ", ".join(
            f"{code}={count}" for code, count in lint_by_code.most_common()
        ))
    lines.append("")

    lines.append("## Bundle candidates (repeated param sets)")
    bundle_lines = _render_bundle_candidates(bundle_counts)
    lines.extend(bundle_lines if bundle_lines else ["- None (no repeated param sets)."])
    lines.append("")

    lines.append("## Param clusters (repeated params)")
    cluster_lines = _render_param_clusters(param_to_surfaces)
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
    root = args.root
    snapshot_dir = _latest_snapshot_dir(root)
    decision_path = args.decision or (snapshot_dir / "decision_snapshot.json")
    lint_path = args.lint or (snapshot_dir / "lint.txt")
    output_path = args.output or (snapshot_dir / "consolidation_report.md")

    decision_obj = json.loads(decision_path.read_text())
    decision_lines = decision_obj.get("decision_surfaces", [])
    value_lines = decision_obj.get("value_decision_surfaces", [])

    decision_surfaces = _parse_surfaces(decision_lines, value_encoded=False)
    value_surfaces = _parse_surfaces(value_lines, value_encoded=True)
    lint_entries = _parse_lint_entries(lint_path.read_text().splitlines())

    _write_consolidation_report(output_path, decision_surfaces, value_surfaces, lint_entries)
    if args.json_output is not None:
        suggestions = _build_suggestions(decision_surfaces, value_surfaces)
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
    return main(["docflow", *_coerce_argv(argv)])


def run_decision_tiers_cli(argv: list[str] | None = None) -> int:
    return main(["decision-tiers", *_coerce_argv(argv)])


def run_consolidation_cli(argv: list[str] | None = None) -> int:
    return main(["consolidation", *_coerce_argv(argv)])


def run_lint_summary_cli(argv: list[str] | None = None) -> int:
    return main(["lint-summary", *_coerce_argv(argv)])


if __name__ == "__main__":
    raise SystemExit(main())
