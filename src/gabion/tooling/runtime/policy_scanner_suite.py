# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import ast
from gabion.tooling.policy_rules import (
    branchless_rule,
    defensive_fallback_rule,
    no_monkeypatch_rule,
    runtime_narrowing_boundary_rule,
    typing_surface_rule,
)
from gabion.tooling.runtime import policy_result_schema

_POLICY_ARTIFACT = Path("artifacts/out/policy_suite_results.json")
_FORMAT_VERSION = 1
_BRANCHLESS_BASELINE = Path("baselines/branchless_policy_baseline.json")
_DEFENSIVE_BASELINE = Path("baselines/defensive_fallback_policy_baseline.json")
_TYPING_SURFACE_BASELINE = Path("baselines/typing_surface_policy_baseline.json")
_TYPING_SURFACE_WAIVERS = Path("baselines/typing_surface_policy_waivers.json")
_RUNTIME_NARROWING_BOUNDARY_BASELINE = Path("baselines/runtime_narrowing_boundary_policy_baseline.json")
_RUNTIME_NARROWING_BOUNDARY_WAIVERS = Path("baselines/runtime_narrowing_boundary_policy_waivers.json")
_LEGACY_MONOLITH_MODULE_PATH = Path("src/gabion/analysis/legacy_dataflow_monolith.py")


_ORCHESTRATOR_PRIMITIVE_BARREL_PATH = Path("src/gabion/server_core/command_orchestrator_primitives.py")
_ORCHESTRATOR_PRIMITIVE_MAX_LINES = 2400
_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS = 220

_EXTERNAL_POLICY_RESULT_RULE_IDS = (
    "policy_check",
    "structural_hash",
    "deprecated_nonerasability",
)


def _normalize_policy_results(raw: object) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key in _EXTERNAL_POLICY_RESULT_RULE_IDS:
        item = raw.get(key)
        if isinstance(item, dict):
            normalized[key] = dict(item)
    return normalized


def _scan_orchestrator_primitive_barrel(*, root: Path) -> list[dict[str, Any]]:
    path = root / _ORCHESTRATOR_PRIMITIVE_BARREL_PATH
    if not path.exists():
        return []
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    export_count = source.count("'" ) if "__all__" in source else 0
    violations: list[dict[str, Any]] = []
    if len(lines) > _ORCHESTRATOR_PRIMITIVE_MAX_LINES:
        violations.append({
            "path": _ORCHESTRATOR_PRIMITIVE_BARREL_PATH.as_posix(),
            "line": 1,
            "column": 1,
            "kind": "line_threshold",
            "message": f"command_orchestrator_primitives.py exceeds line threshold {_ORCHESTRATOR_PRIMITIVE_MAX_LINES}",
            "render": f"{_ORCHESTRATOR_PRIMITIVE_BARREL_PATH.as_posix()}:1:1: line_threshold: exceeds {_ORCHESTRATOR_PRIMITIVE_MAX_LINES} lines",
        })
    if "__all__" in source:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            export_count = len(node.value.elts)
        if export_count > _ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS:
            violations.append({
                "path": _ORCHESTRATOR_PRIMITIVE_BARREL_PATH.as_posix(),
                "line": 1,
                "column": 1,
                "kind": "export_threshold",
                "message": f"command_orchestrator_primitives.py __all__ exports exceed {_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS}",
                "render": f"{_ORCHESTRATOR_PRIMITIVE_BARREL_PATH.as_posix()}:1:1: export_threshold: __all__ exceeds {_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS} symbols",
            })
    return violations


@dataclass(frozen=True)
class PolicySuiteResult:
    root: Path
    inventory_hash: str
    rule_set_hash: str
    violations_by_rule: dict[str, list[dict[str, Any]]]
    policy_results: dict[str, dict[str, Any]]
    cached: bool

    def total_violations(self) -> int:
        return sum(len(items) for items in self.violations_by_rule.values())

    def to_payload(self) -> dict[str, object]:
        counts = {
            rule: len(items)
            for rule, items in self.violations_by_rule.items()
        }
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "inventory_hash": self.inventory_hash,
            "rule_set_hash": self.rule_set_hash,
            "cached": self.cached,
            "counts": counts,
            "violations": self.violations_by_rule,
            "policy_results": self.policy_results,
        }


# gabion:decision_protocol
def load_or_scan_policy_suite(
    *,
    root: Path,
    artifact_path: Path = _POLICY_ARTIFACT,
    policy_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    files = _inventory_files(resolved_root)
    inventory_hash = _inventory_hash(files, resolved_root)
    rule_set_hash = _rule_set_hash()
    normalized_policy_results = {key: dict(value) for key, value in (policy_results or {}).items() if isinstance(value, Mapping)}
    policy_results_hash = hashlib.sha256(json.dumps(normalized_policy_results, sort_keys=True).encode("utf-8")).hexdigest()
    cached_payload = _load_cached_payload(artifact_path)
    if cached_payload is not None:
        if (
            str(cached_payload.get("inventory_hash", "")) == inventory_hash
            and str(cached_payload.get("rule_set_hash", "")) == rule_set_hash
            and str(cached_payload.get("policy_results_hash", "")) == policy_results_hash
        ):
            violations = _violations_from_payload(cached_payload)
            return PolicySuiteResult(
                root=resolved_root,
                inventory_hash=inventory_hash,
                rule_set_hash=rule_set_hash,
                violations_by_rule=violations,
                policy_results=_normalize_policy_results(cached_payload.get("policy_results")),
                cached=True,
            )

    result = scan_policy_suite(root=resolved_root, files=files, policy_results=normalized_policy_results)
    payload = result.to_payload()
    payload["policy_results_hash"] = policy_results_hash
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return result


# gabion:decision_protocol
def scan_policy_suite(
    *,
    root: Path,
    files: tuple[Path, ...] | None = None,
    policy_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    inventory = files if files is not None else _inventory_files(resolved_root)
    inventory_hash = _inventory_hash(inventory, resolved_root)
    rule_set_hash = _rule_set_hash()
    branchless_allowed = _load_rule_baseline_keys(
        module=branchless_rule,
        baseline_path=resolved_root / _BRANCHLESS_BASELINE,
    )
    defensive_allowed = _load_rule_baseline_keys(
        module=defensive_fallback_rule,
        baseline_path=resolved_root / _DEFENSIVE_BASELINE,
    )
    typing_surface_allowed = _load_rule_baseline_keys(
        module=typing_surface_rule,
        baseline_path=resolved_root / _TYPING_SURFACE_BASELINE,
    )
    runtime_narrowing_boundary_allowed = _load_rule_baseline_keys(
        module=runtime_narrowing_boundary_rule,
        baseline_path=resolved_root / _RUNTIME_NARROWING_BOUNDARY_BASELINE,
    )
    typing_surface_waiver_result = typing_surface_rule.load_waivers(
        resolved_root / _TYPING_SURFACE_WAIVERS,
    )
    runtime_narrowing_boundary_waiver_result = runtime_narrowing_boundary_rule.load_waivers(
        resolved_root / _RUNTIME_NARROWING_BOUNDARY_WAIVERS,
    )

    violations_by_rule: dict[str, list[dict[str, Any]]] = {
        "no_monkeypatch": [],
        "branchless": [],
        "defensive_fallback": [],
        "no_legacy_monolith_import": [],
        "orchestrator_primitive_barrel": [],
        "typing_surface": [],
        "runtime_narrowing_boundary": [],
    }

    for invalid in typing_surface_waiver_result.invalid_waivers:
        violations_by_rule["typing_surface"].append(
            {
                "path": _TYPING_SURFACE_WAIVERS.as_posix(),
                "line": int(invalid.index),
                "column": 1,
                "qualname": "<waiver>",
                "kind": "invalid_waiver",
                "scope": "waiver",
                "annotation": "<none>",
                "message": f"invalid typing-surface waiver metadata: {invalid.reason}",
                "key": f"{_TYPING_SURFACE_WAIVERS.as_posix()}:<waiver>:{int(invalid.index)}:invalid_waiver",
                "render": f"{_TYPING_SURFACE_WAIVERS.as_posix()}:{int(invalid.index)}:1: invalid_waiver: {invalid.reason}",
            }
        )

    for invalid in runtime_narrowing_boundary_waiver_result.invalid_waivers:
        violations_by_rule["runtime_narrowing_boundary"].append(
            {
                "path": _RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix(),
                "line": int(invalid.index),
                "column": 1,
                "qualname": "<waiver>",
                "kind": "invalid_waiver",
                "call": "<none>",
                "message": f"invalid runtime-narrowing-boundary waiver metadata: {invalid.reason}",
                "key": f"{_RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix()}:<waiver>:{int(invalid.index)}:invalid_waiver",
                "render": f"{_RUNTIME_NARROWING_BOUNDARY_WAIVERS.as_posix()}:{int(invalid.index)}:1: invalid_waiver: {invalid.reason}",
            }
        )

    legacy_module_path = resolved_root / _LEGACY_MONOLITH_MODULE_PATH
    if legacy_module_path.exists():
        violations_by_rule["no_legacy_monolith_import"].append(
            _serialize_legacy_monolith(
                _LegacyMonolithViolation(
                    path=_LEGACY_MONOLITH_MODULE_PATH.as_posix(),
                    line=1,
                    column=1,
                    kind="module_present",
                    message="retired legacy monolith module must not be present",
                )
            )
        )

    violations_by_rule["orchestrator_primitive_barrel"].extend(
        _scan_orchestrator_primitive_barrel(root=resolved_root)
    )

    for path in inventory:
        rel_path = path.relative_to(resolved_root).as_posix()
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = _parse_tree(source, rel_path=rel_path)
        if tree is not None:
            if rel_path.startswith("src/") or rel_path.startswith("tests/"):
                no_legacy_monolith = _NoLegacyMonolithVisitor(rel_path=rel_path)
                no_legacy_monolith.visit(tree)
                violations_by_rule["no_legacy_monolith_import"].extend(
                    _serialize_legacy_monolith(item) for item in no_legacy_monolith.violations
                )

            if rel_path.startswith("src/") or rel_path.startswith("tests/"):
                no_mp = no_monkeypatch_rule._NoMonkeypatchVisitor(rel_path=rel_path)
                no_mp.visit(tree)
                violations_by_rule["no_monkeypatch"].extend(
                    _serialize_no_monkeypatch(item) for item in no_mp.violations
                )

            if rel_path.startswith("src/gabion/"):
                branchless_visitor = branchless_rule._BranchlessVisitor(
                    rel_path=rel_path,
                    source_lines=source_lines,
                )
                branchless_visitor.visit(tree)
                branchless_violations = _filter_baseline_violations(
                    branchless_visitor.violations,
                    allowed_keys=branchless_allowed,
                )
                violations_by_rule["branchless"].extend(
                    _serialize_branchless(item) for item in branchless_violations
                )

                defensive_visitor = defensive_fallback_rule._DefensiveFallbackVisitor(
                    rel_path=rel_path,
                    source_lines=source_lines,
                )
                defensive_visitor.visit(tree)
                defensive_violations = _filter_baseline_violations(
                    defensive_visitor.violations,
                    allowed_keys=defensive_allowed,
                )
                violations_by_rule["defensive_fallback"].extend(
                    _serialize_defensive(item) for item in defensive_violations
                )

                typing_surface_violations = _filter_baseline_violations(
                    typing_surface_rule.collect_violations(
                        rel_path=rel_path,
                        source=source,
                        tree=tree,
                    ),
                    allowed_keys=(typing_surface_allowed | typing_surface_waiver_result.allowed_keys),
                )
                violations_by_rule["typing_surface"].extend(
                    _serialize_typing_surface(item) for item in typing_surface_violations
                )

                runtime_narrowing_boundary_violations = _filter_baseline_violations(
                    runtime_narrowing_boundary_rule.collect_violations(
                        rel_path=rel_path,
                        source=source,
                        tree=tree,
                    ),
                    allowed_keys=(runtime_narrowing_boundary_allowed | runtime_narrowing_boundary_waiver_result.allowed_keys),
                )
                violations_by_rule["runtime_narrowing_boundary"].extend(
                    _serialize_runtime_narrowing_boundary(item)
                    for item in runtime_narrowing_boundary_violations
                )

    for rule, items in list(violations_by_rule.items()):
        violations_by_rule[rule] = sorted(
            items,
            key=lambda item: (
                str(item.get("path", "")),
                str(item.get("qualname", "")),
                int(item.get("line", 0) or 0),
                str(item.get("kind", "")),
            ),
        )
    return PolicySuiteResult(
        root=resolved_root,
        inventory_hash=inventory_hash,
        rule_set_hash=rule_set_hash,
        violations_by_rule=violations_by_rule,
        policy_results={key: dict(value) for key, value in (policy_results or {}).items() if isinstance(value, Mapping)},
        cached=False,
    )


def violations_for_rule(result: PolicySuiteResult, *, rule: str) -> list[dict[str, Any]]:
    return list(result.violations_by_rule.get(rule, []))


def _load_cached_payload(path: Path) -> dict[str, object] | None:
    payload: dict[str, object] | None = None
    if path.exists():
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_payload = None
        if isinstance(raw_payload, dict):
            format_version = int(raw_payload.get("format_version", 0) or 0)
            if format_version == _FORMAT_VERSION:
                payload = raw_payload
    return payload


def _violations_from_payload(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    violations_raw = payload.get("violations")
    if not isinstance(violations_raw, dict):
        return {
            "no_monkeypatch": [],
            "branchless": [],
            "defensive_fallback": [],
            "no_legacy_monolith_import": [],
            "orchestrator_primitive_barrel": [],
            "typing_surface": [],
            "runtime_narrowing_boundary": [],
        }
    normalized: dict[str, list[dict[str, Any]]] = {}
    for rule in (
        "no_monkeypatch",
        "branchless",
        "defensive_fallback",
        "no_legacy_monolith_import",
        "orchestrator_primitive_barrel",
        "typing_surface",
        "runtime_narrowing_boundary",
    ):
        raw_items = violations_raw.get(rule)
        if not isinstance(raw_items, list):
            normalized[rule] = []
            continue
        normalized[rule] = [dict(item) for item in raw_items if isinstance(item, dict)]
    return normalized


def _inventory_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for pattern in ("src/gabion/**/*.py", "tests/**/*.py"):
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or any(part == "__pycache__" for part in path.parts):
                continue
            files.append(path.resolve())
    deduped = sorted(set(files), key=lambda item: str(item))
    return tuple(deduped)


def _inventory_hash(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        rel = path.relative_to(root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
        digest.update(str(int(stat.st_size)).encode("utf-8"))
    return digest.hexdigest()


def _rule_set_hash() -> str:
    material = "|".join(
        [
            "no_monkeypatch:v1",
            "branchless:v2",
            "defensive_fallback:v2",
            "no_legacy_monolith_import:v1",
            "orchestrator_primitive_barrel:v1",
            "typing_surface:v2",
            "runtime_narrowing_boundary:v2",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _parse_tree(source: str, *, rel_path: str):
    try:
        return ast.parse(source)
    except SyntaxError:
        # Surface syntax failures through existing rule scripts instead of reclassifying here.
        return None


@dataclass(frozen=True)
class _LegacyMonolithViolation:
    path: str
    line: int
    column: int
    kind: str
    message: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.line}:{self.column}:{self.kind}:{self.message}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.kind}: {self.message}"


class _NoLegacyMonolithVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self._path = rel_path
        self.violations: list[_LegacyMonolithViolation] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name
            if module_name == "gabion.analysis.legacy_dataflow_monolith":
                self._record(
                    node,
                    kind="import",
                    message="legacy_dataflow_monolith import is retired; use owned modules only",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        has_direct_legacy_alias = any(alias.name == "legacy_dataflow_monolith" for alias in node.names)
        if module_name == "gabion.analysis.legacy_dataflow_monolith":
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif module_name == "gabion.analysis" and has_direct_legacy_alias:
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif node.level > 0 and module_name.endswith("legacy_dataflow_monolith"):
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        elif node.level > 0 and module_name == "" and has_direct_legacy_alias:
            self._record(
                node,
                kind="import_from",
                message="legacy_dataflow_monolith import is retired; use owned modules only",
            )
        self.generic_visit(node)

    def _record(self, node: ast.AST, *, kind: str, message: str) -> None:
        self.violations.append(
            _LegacyMonolithViolation(
                path=self._path,
                line=int(getattr(node, "lineno", 1) or 1),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
                kind=kind,
                message=message,
            )
        )


def _load_rule_baseline_keys(*, module: object, baseline_path: Path) -> set[str]:
    loader = getattr(module, "_load_baseline", None)
    if loader is None:
        return set()
    try:
        loaded = loader(baseline_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return set()
    if not isinstance(loaded, set):
        return set()
    return {str(item) for item in loaded if isinstance(item, str)}


def _filter_baseline_violations(
    violations: Iterable[object],
    *,
    allowed_keys: set[str],
) -> list[object]:
    if not allowed_keys:
        return list(violations)
    filtered: list[object] = []
    for violation in violations:
        key = str(getattr(violation, "key", "") or "")
        legacy_key = str(getattr(violation, "legacy_key", "") or "")
        if (key and key in allowed_keys) or (legacy_key and legacy_key in allowed_keys):
            continue
        filtered.append(violation)
    return filtered


def _serialize_no_monkeypatch(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "message": getattr(violation, "message"),
        "render": getattr(violation, "render")(),
    }


def _serialize_branchless(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_defensive(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_legacy_monolith(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_typing_surface(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "scope": getattr(violation, "scope"),
        "annotation": getattr(violation, "annotation"),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_runtime_narrowing_boundary(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


__all__ = [
    "PolicySuiteResult",
    "load_or_scan_policy_suite",
    "scan_policy_suite",
    "violations_for_rule",
]
