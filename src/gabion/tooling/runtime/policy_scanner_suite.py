# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
import ast
from gabion.tooling.policy_rules import branchless_rule, defensive_fallback_rule, no_monkeypatch_rule

_POLICY_ARTIFACT = Path("artifacts/out/policy_suite_results.json")
_FORMAT_VERSION = 1
_BRANCHLESS_BASELINE = Path("baselines/branchless_policy_baseline.json")
_DEFENSIVE_BASELINE = Path("baselines/defensive_fallback_policy_baseline.json")
_TYPE_DEBT_BASELINE = Path("baselines/type_contract_debt_baseline.json")
_LEGACY_MONOLITH_MODULE_PATH = Path("src/gabion/analysis/legacy_dataflow_monolith.py")


_ORCHESTRATOR_PRIMITIVE_BARREL_PATH = Path("src/gabion/server_core/command_orchestrator_primitives.py")
_ORCHESTRATOR_PRIMITIVE_MAX_LINES = 2400
_ORCHESTRATOR_PRIMITIVE_MAX_ALL_SYMBOLS = 220
_TYPE_DEBT_RULE = "type_contract_debt"
_TYPE_DEBT_KINDS = (
    "any_occurrence",
    "public_bare_object_contract",
    "non_boundary_dict_str_object_signature",
    "narrowing_operator_outside_boundary",
)
_TYPE_DEBT_BOUNDARY_MARKERS = (
    "gabion:boundary_normalization",
    "gabion:ambiguity_boundary_module",
    "gabion:ambiguity_boundary",
)
_TYPE_DEBT_APPROVED_BOUNDARY_MODULE_TOKENS = ("/boundary", "_boundary")


def _scan_orchestrator_primitive_barrel(*, root: Path) -> list[dict[str, object]]:
    path = root / _ORCHESTRATOR_PRIMITIVE_BARREL_PATH
    if not path.exists():
        return []
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    export_count = source.count("'" ) if "__all__" in source else 0
    violations: list[dict[str, object]] = []
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
    violations_by_rule: dict[str, list[dict[str, object]]]
    cached: bool

    def total_violations(self) -> int:
        return sum(len(items) for items in self.violations_by_rule.values())

    def to_payload(self) -> dict[str, object]:
        counts = {
            rule: len(items)
            for rule, items in self.violations_by_rule.items()
        }
        type_debt_counts = {kind: 0 for kind in _TYPE_DEBT_KINDS}
        for item in self.violations_by_rule.get(_TYPE_DEBT_RULE, []):
            kind = str(item.get("kind", "") or "")
            if kind in type_debt_counts:
                type_debt_counts[kind] += 1
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "inventory_hash": self.inventory_hash,
            "rule_set_hash": self.rule_set_hash,
            "cached": self.cached,
            "counts": counts,
            "type_contract_debt_counts": type_debt_counts,
            "violations": self.violations_by_rule,
        }


# gabion:decision_protocol
def load_or_scan_policy_suite(
    *,
    root: Path,
    artifact_path: Path = _POLICY_ARTIFACT,
) -> PolicySuiteResult:
    resolved_root = root.resolve()
    files = _inventory_files(resolved_root)
    inventory_hash = _inventory_hash(files, resolved_root)
    rule_set_hash = _rule_set_hash()
    cached_payload = _load_cached_payload(artifact_path)
    if cached_payload is not None:
        if (
            str(cached_payload.get("inventory_hash", "")) == inventory_hash
            and str(cached_payload.get("rule_set_hash", "")) == rule_set_hash
        ):
            violations = _violations_from_payload(cached_payload)
            return PolicySuiteResult(
                root=resolved_root,
                inventory_hash=inventory_hash,
                rule_set_hash=rule_set_hash,
                violations_by_rule=violations,
                cached=True,
            )

    result = scan_policy_suite(root=resolved_root, files=files)
    payload = result.to_payload()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return result


# gabion:decision_protocol
def scan_policy_suite(*, root: Path, files: tuple[Path, ...] | None = None) -> PolicySuiteResult:
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
    type_debt_baseline = _load_type_debt_baseline_payload(resolved_root / _TYPE_DEBT_BASELINE)

    violations_by_rule: dict[str, list[dict[str, object]]] = {
        "no_monkeypatch": [],
        "branchless": [],
        "defensive_fallback": [],
        "no_legacy_monolith_import": [],
        "orchestrator_primitive_barrel": [],
        _TYPE_DEBT_RULE: [],
    }

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
                type_debt_visitor = _TypeContractDebtVisitor(
                    rel_path=rel_path,
                    source_lines=source_lines,
                )
                type_debt_visitor.visit(tree)
                violations_by_rule[_TYPE_DEBT_RULE].extend(
                    _serialize_type_debt(item) for item in type_debt_visitor.violations
                )

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

    violations_by_rule[_TYPE_DEBT_RULE].extend(
        _ratchet_type_debt(
            findings=violations_by_rule[_TYPE_DEBT_RULE],
            baseline=type_debt_baseline.thresholds,
            waived_kinds=type_debt_baseline.waivers,
        )
    )
    violations_by_rule[_TYPE_DEBT_RULE] = sorted(
        violations_by_rule[_TYPE_DEBT_RULE],
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
        cached=False,
    )


def violations_for_rule(result: PolicySuiteResult, *, rule: str) -> list[dict[str, object]]:
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


def _violations_from_payload(payload: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    violations_raw = payload.get("violations")
    if not isinstance(violations_raw, dict):
        return {
            "no_monkeypatch": [],
            "branchless": [],
            "defensive_fallback": [],
            "no_legacy_monolith_import": [],
            "orchestrator_primitive_barrel": [],
            _TYPE_DEBT_RULE: [],
        }
    normalized: dict[str, list[dict[str, object]]] = {}
    for rule in ("no_monkeypatch", "branchless", "defensive_fallback", "no_legacy_monolith_import", "orchestrator_primitive_barrel", _TYPE_DEBT_RULE):
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
            "branchless:v1",
            "defensive_fallback:v1",
            "no_legacy_monolith_import:v1",
            "orchestrator_primitive_barrel:v1",
            "type_contract_debt:v1",
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


@dataclass(frozen=True)
class _TypeDebtViolation:
    path: str
    line: int
    column: int
    kind: str
    message: str
    qualname: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.path}:{self.qualname}:{self.line}:{self.column}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.kind}] [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _TypeDebtScope:
    qualname: str
    is_boundary: bool


@dataclass(frozen=True)
class _TypeDebtBaseline:
    thresholds: dict[str, int]
    waivers: set[str]


class _TypeContractDebtVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str, source_lines: list[str]) -> None:
        self._path = rel_path
        self._source_lines = source_lines
        self._module_boundary = _module_is_type_debt_boundary(rel_path=rel_path, source_lines=source_lines)
        self._scopes: list[_TypeDebtScope] = []
        self.violations: list[_TypeDebtViolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._scan_annotation(annotation=node.annotation, node=node, public_contract=False)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._scope_is_boundary:
            self.generic_visit(node)
            return
        if isinstance(node.func, ast.Name) and node.func.id in {"isinstance", "cast"}:
            self._record(
                node,
                kind="narrowing_operator_outside_boundary",
                message=f"{node.func.id} narrowing must be isolated to approved boundary modules",
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "cast":
            dotted = _dotted_name(node.func.value)
            if dotted in {"typing", "typing_extensions"}:
                self._record(
                    node,
                    kind="narrowing_operator_outside_boundary",
                    message="cast narrowing must be isolated to approved boundary modules",
                )
        self.generic_visit(node)

    @property
    def _scope_is_boundary(self) -> bool:
        if self._scopes:
            return self._scopes[-1].is_boundary
        return self._module_boundary

    @property
    def _scope_qualname(self) -> str:
        if self._scopes:
            return self._scopes[-1].qualname
        return "<module>"

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._scope_qualname
        qualname = node.name if parent == "<module>" else f"{parent}.{node.name}"
        is_boundary = self._module_boundary or _has_boundary_marker(
            source_lines=self._source_lines,
            line=int(getattr(node, "lineno", 1) or 1),
        )
        self._scopes.append(_TypeDebtScope(qualname=qualname, is_boundary=is_boundary))
        for arg in _iter_function_args(node):
            if arg.annotation is not None:
                self._scan_annotation(annotation=arg.annotation, node=arg, public_contract=_is_public_name(node.name))
        if node.returns is not None:
            self._scan_annotation(annotation=node.returns, node=node, public_contract=_is_public_name(node.name))
        self.generic_visit(node)
        self._scopes.pop()

    def _scan_annotation(self, *, annotation: ast.AST, node: ast.AST, public_contract: bool) -> None:
        if _annotation_contains_name(annotation, "Any"):
            self._record(node, kind="any_occurrence", message="Any in type contract")
        if public_contract and _annotation_is_bare_object(annotation):
            self._record(
                node,
                kind="public_bare_object_contract",
                message="public contract uses bare object type",
            )
        if not self._scope_is_boundary and _annotation_is_dict_str_object(annotation):
            self._record(
                node,
                kind="non_boundary_dict_str_object_signature",
                message="dict[str, object] signature must be normalized at a boundary",
            )

    def _record(self, node: ast.AST, *, kind: str, message: str) -> None:
        self.violations.append(
            _TypeDebtViolation(
                path=self._path,
                line=int(getattr(node, "lineno", 1) or 1),
                column=int(getattr(node, "col_offset", 0) or 0) + 1,
                kind=kind,
                message=message,
                qualname=self._scope_qualname,
            )
        )



def _module_is_type_debt_boundary(*, rel_path: str, source_lines: list[str]) -> bool:
    normalized = f"/{rel_path.lower()}"
    if any(token in normalized for token in _TYPE_DEBT_APPROVED_BOUNDARY_MODULE_TOKENS):
        return True
    return _has_boundary_marker(source_lines=source_lines, line=1)


def _has_boundary_marker(*, source_lines: list[str], line: int) -> bool:
    idx = max(0, line - 2)
    while idx >= 0:
        stripped = source_lines[idx].strip()
        if not stripped:
            idx -= 1
            continue
        if not stripped.startswith("#"):
            return False
        return any(marker in stripped for marker in _TYPE_DEBT_BOUNDARY_MARKERS)
    return False


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _iter_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)
    return args


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _annotation_contains_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


def _annotation_is_bare_object(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "object"


def _annotation_is_dict_str_object(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    target = _dotted_name(node.value)
    if target not in {"dict", "typing.Dict", "Dict"}:
        return False
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple) and len(slice_node.elts) == 2:
        left, right = slice_node.elts
        return _annotation_is_str_type(left) and _annotation_is_object_type(right)
    return False


def _annotation_is_str_type(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "str") or (
        isinstance(node, ast.Constant) and node.value == "str"
    )


def _annotation_is_object_type(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == "object") or (
        isinstance(node, ast.Constant) and node.value == "object"
    )


def _load_type_debt_baseline_payload(path: Path) -> _TypeDebtBaseline:
    defaults = {kind: 0 for kind in _TYPE_DEBT_KINDS}
    if not path.exists():
        return _TypeDebtBaseline(thresholds=defaults, waivers=set())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _TypeDebtBaseline(thresholds=defaults, waivers=set())
    thresholds_raw = payload.get("thresholds") if isinstance(payload, dict) else None
    waivers_raw = payload.get("waivers") if isinstance(payload, dict) else None
    merged = dict(defaults)
    if isinstance(thresholds_raw, dict):
        for kind in _TYPE_DEBT_KINDS:
            value = thresholds_raw.get(kind, 0)
            merged[kind] = int(value) if isinstance(value, int) and value >= 0 else 0
    waivers: set[str] = set()
    if isinstance(waivers_raw, dict):
        for kind, note in waivers_raw.items():
            if kind in _TYPE_DEBT_KINDS and isinstance(note, str) and note.strip():
                waivers.add(kind)
    return _TypeDebtBaseline(thresholds=merged, waivers=waivers)

def _ratchet_type_debt(*, findings: list[dict[str, object]], baseline: dict[str, int], waived_kinds: set[str]) -> list[dict[str, object]]:
    counts = {kind: 0 for kind in _TYPE_DEBT_KINDS}
    for item in findings:
        kind = str(item.get("kind", "") or "")
        if kind in counts:
            counts[kind] += 1
    ratchet: list[dict[str, object]] = []
    for kind in _TYPE_DEBT_KINDS:
        if kind in waived_kinds:
            continue
        current = int(counts.get(kind, 0))
        allowed = int(baseline.get(kind, 0))
        if current <= allowed:
            continue
        ratchet.append(
            {
                "path": _TYPE_DEBT_BASELINE.as_posix(),
                "line": 1,
                "column": 1,
                "kind": "ratchet_regression",
                "qualname": kind,
                "message": f"type-contract debt for {kind} increased above baseline ({current}>{allowed})",
                "key": f"ratchet:{kind}:{current}:{allowed}",
                "render": f"{_TYPE_DEBT_BASELINE.as_posix()}:1:1: [ratchet_regression] [{kind}] current={current} baseline={allowed}",
                "current": current,
                "baseline": allowed,
            }
        )
    return ratchet


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
        if key and key in allowed_keys:
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


def _serialize_type_debt(violation: object) -> dict[str, object]:
    return {
        "path": getattr(violation, "path"),
        "line": getattr(violation, "line"),
        "column": getattr(violation, "column"),
        "qualname": getattr(violation, "qualname"),
        "kind": getattr(violation, "kind"),
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
