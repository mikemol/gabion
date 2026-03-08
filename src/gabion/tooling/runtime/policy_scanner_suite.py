# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping
import ast
from gabion.tooling.policy_rules import (
    aspf_normalization_idempotence_rule,
    boundary_core_contract_rule,
    branchless_rule,
    defensive_fallback_rule,
    fiber_normalization_contract_rule,
    fiber_scalar_sentinel_contract_rule,
    no_monkeypatch_rule,
    runtime_narrowing_boundary_rule,
    test_sleep_hygiene_rule,
    test_subprocess_hygiene_rule,
    typing_surface_rule,
)
from gabion.tooling.policy_rules.fiber_diagnostics import (
    to_payload_bounds as _fiber_bounds_payload,
    to_payload_counterfactual as _fiber_counterfactual_payload,
    to_payload_trace as _fiber_trace_payload,
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
_ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE = Path("baselines/aspf_normalization_idempotence_policy_baseline.json")
_LEGACY_MONOLITH_MODULE_PATH = Path("src/gabion/analysis/legacy_dataflow_monolith.py")
_TEST_SUBPROCESS_HYGIENE_ALLOWLIST = Path("docs/policy/test_subprocess_hygiene_allowlist.txt")
_TEST_SLEEP_HYGIENE_ALLOWLIST = Path("docs/policy/test_sleep_hygiene_allowlist.txt")


_ORCHESTRATOR_PRIMITIVE_BARREL_PATH = Path("src/gabion/server_core/command_orchestrator_primitives.py")
_ORCHESTRATOR_PRIMITIVE_MAX_LINES = 2400
_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS = 220

_EXTERNAL_POLICY_RESULT_RULE_IDS = (
    "policy_check",
    "structural_hash",
    "deprecated_nonerasability",
)
_BOUNDARY_MARKER = "gabion:boundary_normalization_module"


def _normalize_policy_results(raw: object) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    match raw:
        case dict() as payload:
            for key in _EXTERNAL_POLICY_RESULT_RULE_IDS:
                item = payload.get(key)
                match item:
                    case dict() as item_payload:
                        normalized[key] = dict(item_payload)
                    case _:
                        pass
        case _:
            pass
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
                match node:
                    case ast.Assign(targets=targets, value=ast.List(elts=elts)):
                        has_all_target = any(
                            _assign_target_name(target) == "__all__"
                            for target in targets
                        )
                        if has_all_target:
                            export_count = len(elts)
                    case ast.Assign(targets=targets, value=ast.Tuple(elts=elts)):
                        has_all_target = any(
                            _assign_target_name(target) == "__all__"
                            for target in targets
                        )
                        if has_all_target:
                            export_count = len(elts)
                    case _:
                        pass
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


def _changed_paths_from_git(
    *,
    root: Path,
    base_sha: str | None,
    head_sha: str | None,
) -> set[str] | None:
    if base_sha and head_sha:
        command = ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"]
    else:
        command = ["git", "diff", "--name-only", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    changed = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if base_sha and head_sha:
        return changed

    # Include untracked files for local touched+new checks.
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return changed
    changed.update(line.strip() for line in untracked.stdout.splitlines() if line.strip())
    return changed


def _boundary_scoped_files(
    *,
    root: Path,
    inventory: tuple[Path, ...],
    changed_paths: set[str] | None,
) -> tuple[Path, ...]:
    scoped: list[Path] = []
    for path in inventory:
        rel = path.relative_to(root).as_posix()
        path_in_scope = changed_paths is None or rel in changed_paths
        path_under_src = rel.startswith("src/gabion/")
        if path_in_scope and path_under_src:
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                source = None
            if source is not None and _BOUNDARY_MARKER in source:
                scoped.append(path)
    return tuple(sorted(set(scoped), key=lambda item: str(item)))


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
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    files = _inventory_files(resolved_root)
    inventory_hash = _inventory_hash(files, resolved_root)
    changed_paths = _changed_paths_from_git(
        root=resolved_root,
        base_sha=base_sha,
        head_sha=head_sha,
    )
    changed_scope_hash = hashlib.sha256(
        json.dumps(sorted(changed_paths) if changed_paths is not None else ["<all>"]).encode(
            "utf-8"
        )
    ).hexdigest()
    rule_set_hash = _rule_set_hash()
    normalized_policy_results = _normalized_policy_result_mapping(
        policy_results or {},
    )
    policy_results_hash = hashlib.sha256(json.dumps(normalized_policy_results, sort_keys=True).encode("utf-8")).hexdigest()
    cached_payload = _load_cached_payload(artifact_path)
    if cached_payload is not None:
        if (
            str(cached_payload.get("inventory_hash", "")) == inventory_hash
            and str(cached_payload.get("rule_set_hash", "")) == rule_set_hash
            and str(cached_payload.get("policy_results_hash", "")) == policy_results_hash
            and str(cached_payload.get("changed_scope_hash", "")) == changed_scope_hash
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

    result = scan_policy_suite(
        root=resolved_root,
        files=files,
        policy_results=normalized_policy_results,
        base_sha=base_sha,
        head_sha=head_sha,
        changed_paths=changed_paths,
    )
    payload = result.to_payload()
    payload["policy_results_hash"] = policy_results_hash
    payload["changed_scope_hash"] = changed_scope_hash
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return result


# gabion:decision_protocol
def scan_policy_suite(
    *,
    root: Path,
    files: tuple[Path, ...] | None = None,
    policy_results: Mapping[str, Mapping[str, Any]] | None = None,
    base_sha: str | None = None,
    head_sha: str | None = None,
    changed_paths: set[str] | None = None,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    inventory = files if files is not None else _inventory_files(resolved_root)
    resolved_changed_paths = (
        changed_paths
        if changed_paths is not None
        else _changed_paths_from_git(
            root=resolved_root,
            base_sha=base_sha,
            head_sha=head_sha,
        )
    )
    boundary_scope_files = _boundary_scoped_files(
        root=resolved_root,
        inventory=inventory,
        changed_paths=resolved_changed_paths,
    )
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
    aspf_normalization_idempotence_allowed = _load_rule_baseline_keys(
        module=aspf_normalization_idempotence_rule,
        baseline_path=resolved_root / _ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE,
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
        "fiber_scalar_sentinel_contract": [],
        "no_legacy_monolith_import": [],
        "orchestrator_primitive_barrel": [],
        "typing_surface": [],
        "runtime_narrowing_boundary": [],
        "aspf_normalization_idempotence": [],
        "boundary_core_contract": [],
        "fiber_normalization_contract": [],
        "test_subprocess_hygiene": [],
        "test_sleep_hygiene": [],
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
    aspf_normalization_idempotence_violations = _filter_baseline_violations(
        aspf_normalization_idempotence_rule.collect_violations(root=resolved_root),
        allowed_keys=aspf_normalization_idempotence_allowed,
    )
    aspf_ingress_violations = aspf_normalization_idempotence_rule.collect_ingress_violations(
        root=resolved_root,
        baseline_path=(resolved_root / _ASPF_NORMALIZATION_IDEMPOTENCE_BASELINE),
    )
    violations_by_rule["aspf_normalization_idempotence"].extend(
        _serialize_aspf_normalization_idempotence(item)
        for item in aspf_ingress_violations
    )
    violations_by_rule["aspf_normalization_idempotence"].extend(
        _serialize_aspf_normalization_idempotence(item)
        for item in aspf_normalization_idempotence_violations
    )
    boundary_core_contract_violations = boundary_core_contract_rule.collect_violations(
        root=resolved_root,
        files=boundary_scope_files,
    )
    violations_by_rule["boundary_core_contract"].extend(
        _serialize_boundary_core_contract(item)
        for item in boundary_core_contract_violations
    )
    fiber_contract_violations = fiber_normalization_contract_rule.collect_violations(
        root=resolved_root,
        files=boundary_scope_files,
    )
    violations_by_rule["fiber_normalization_contract"].extend(
        _serialize_fiber_normalization_contract(item)
        for item in fiber_contract_violations
    )
    test_subprocess_hygiene_violations = test_subprocess_hygiene_rule.collect_violations(
        root=resolved_root,
        allowlist_path=resolved_root / _TEST_SUBPROCESS_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_subprocess_hygiene"].extend(
        _serialize_test_subprocess_hygiene(item)
        for item in test_subprocess_hygiene_violations
    )
    test_sleep_hygiene_violations = test_sleep_hygiene_rule.collect_violations(
        root=resolved_root,
        allowlist_path=resolved_root / _TEST_SLEEP_HYGIENE_ALLOWLIST,
    )
    violations_by_rule["test_sleep_hygiene"].extend(
        _serialize_test_sleep_hygiene(item)
        for item in test_sleep_hygiene_violations
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
                scalar_sentinel_violations = (
                    fiber_scalar_sentinel_contract_rule.collect_violations(
                        rel_path=rel_path,
                        source=source,
                        tree=tree,
                    )
                )
                violations_by_rule["fiber_scalar_sentinel_contract"].extend(
                    _serialize_fiber_scalar_sentinel_contract(item)
                    for item in scalar_sentinel_violations
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
        policy_results=_normalized_policy_result_mapping(policy_results or {}),
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
        match raw_payload:
            case dict() as payload_dict:
                format_version = int(payload_dict.get("format_version", 0) or 0)
                if format_version == _FORMAT_VERSION:
                    payload = payload_dict
            case _:
                pass
    return payload


def _violations_from_payload(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    violations_raw = payload.get("violations")
    match violations_raw:
        case dict():
            pass
        case _:
            return {
                "no_monkeypatch": [],
                "branchless": [],
                "defensive_fallback": [],
                "fiber_scalar_sentinel_contract": [],
                "no_legacy_monolith_import": [],
                "orchestrator_primitive_barrel": [],
                "typing_surface": [],
                "runtime_narrowing_boundary": [],
                "aspf_normalization_idempotence": [],
                "boundary_core_contract": [],
                "fiber_normalization_contract": [],
                "test_subprocess_hygiene": [],
                "test_sleep_hygiene": [],
            }
    normalized: dict[str, list[dict[str, Any]]] = {}
    for rule in (
        "no_monkeypatch",
        "branchless",
        "defensive_fallback",
        "fiber_scalar_sentinel_contract",
        "no_legacy_monolith_import",
        "orchestrator_primitive_barrel",
        "typing_surface",
        "runtime_narrowing_boundary",
        "aspf_normalization_idempotence",
        "boundary_core_contract",
        "fiber_normalization_contract",
        "test_subprocess_hygiene",
        "test_sleep_hygiene",
    ):
        raw_items = violations_raw.get(rule)
        match raw_items:
            case list():
                items: list[dict[str, Any]] = []
                for item in raw_items:
                    match item:
                        case dict() as violation_item:
                            items.append(dict(violation_item))
                        case _:
                            pass
                normalized[rule] = items
            case _:
                normalized[rule] = []
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
            "fiber_scalar_sentinel_contract:v1",
            "no_legacy_monolith_import:v1",
            "orchestrator_primitive_barrel:v1",
            "typing_surface:v2",
            "runtime_narrowing_boundary:v2",
            "aspf_normalization_idempotence:v3",
            "boundary_core_contract:v1",
            "fiber_normalization_contract:v2",
            "test_subprocess_hygiene:v2",
            "test_sleep_hygiene:v1",
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
    match loaded:
        case set() as keys:
            normalized: set[str] = set()
            for item in keys:
                match item:
                    case str() as text:
                        normalized.add(text)
                    case _:
                        pass
            return normalized
        case _:
            return set()


def _assign_target_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case _:
            return None


def _normalized_policy_result_mapping(
    payload: Mapping[str, Mapping[str, Any]] | object,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    try:
        items = payload.items()  # type: ignore[attr-defined]
    except AttributeError:
        return normalized
    for key, value in items:
        mapping = _mapping_copy_or_none(value)
        if mapping is not None:
            normalized[str(key)] = mapping
    return normalized


def _mapping_copy_or_none(value: object) -> dict[str, Any] | None:
    try:
        mapping = dict(value.items())  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return None
    return mapping


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


def _serialize_fiber_scalar_sentinel_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "scalar_literal": getattr(violation, "scalar_literal"),
        "comparison_operator": getattr(violation, "comparison_operator"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _fiber_bounds_payload(
            getattr(violation, "applicability_bounds", None)
        )
        if getattr(violation, "applicability_bounds", None) is not None
        else None,
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
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


def _serialize_aspf_normalization_idempotence(
    violation: object,
) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "normalization_class": getattr(violation, "normalization_class"),
        "flow_identity": getattr(violation, "flow_identity"),
        "event_kind": getattr(violation, "event_kind"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _fiber_bounds_payload(
            getattr(violation, "applicability_bounds", None)
        )
        if getattr(violation, "applicability_bounds", None) is not None
        else None,
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "message": getattr(violation, "message"),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_boundary_core_contract(violation: object) -> dict[str, object]:
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


def _serialize_fiber_normalization_contract(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
        "message": getattr(violation, "message"),
        "normalization_class": getattr(violation, "normalization_class"),
        "input_slot": getattr(violation, "input_slot"),
        "flow_identity": getattr(violation, "flow_identity"),
        "fiber_trace": _fiber_trace_payload(
            getattr(violation, "fiber_trace", ())
        ),
        "applicability_bounds": _fiber_bounds_payload(
            getattr(violation, "applicability_bounds", None)
        )
        if getattr(violation, "applicability_bounds", None) is not None
        else None,
        "counterfactual_boundary": _fiber_counterfactual_payload(
            getattr(violation, "counterfactual_boundary", None)
        ),
        "structured_hash": getattr(violation, "structured_hash", ""),
        "legacy_key": getattr(violation, "legacy_key", ""),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_test_subprocess_hygiene(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


def _serialize_test_sleep_hygiene(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "kind": getattr(violation, "kind"),
        "call": getattr(violation, "call"),
        "message": getattr(violation, "message"),
        "key": getattr(violation, "key"),
        "render": getattr(violation, "render")(),
    }


__all__ = [
    "PolicySuiteResult",
    "load_or_scan_policy_suite",
    "scan_policy_suite",
    "violations_for_rule",
]
