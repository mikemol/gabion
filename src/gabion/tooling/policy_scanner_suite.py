# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script_module(module_stem: str):
    module_path = _REPO_ROOT / "scripts" / f"{module_stem}.py"
    spec = importlib.util.spec_from_file_location(f"gabion_policy_{module_stem}", module_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


branchless_policy_check = _load_script_module("branchless_policy_check")
defensive_fallback_policy_check = _load_script_module("defensive_fallback_policy_check")
no_monkeypatch_policy_check = _load_script_module("no_monkeypatch_policy_check")

_POLICY_ARTIFACT = Path("artifacts/out/policy_suite_results.json")
_FORMAT_VERSION = 1


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
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "inventory_hash": self.inventory_hash,
            "rule_set_hash": self.rule_set_hash,
            "cached": self.cached,
            "counts": counts,
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

    violations_by_rule: dict[str, list[dict[str, object]]] = {
        "no_monkeypatch": [],
        "branchless": [],
        "defensive_fallback": [],
    }
    for path in inventory:
        rel_path = path.relative_to(resolved_root).as_posix()
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        tree = _parse_tree(source, rel_path=rel_path)
        if tree is not None:
            if rel_path.startswith("src/") or rel_path.startswith("tests/"):
                no_mp = no_monkeypatch_policy_check._NoMonkeypatchVisitor(rel_path=rel_path)
                no_mp.visit(tree)
                violations_by_rule["no_monkeypatch"].extend(
                    _serialize_no_monkeypatch(item) for item in no_mp.violations
                )

            if rel_path.startswith("src/gabion/"):
                branchless_visitor = branchless_policy_check._BranchlessVisitor(
                    rel_path=rel_path,
                    source_lines=source_lines,
                )
                branchless_visitor.visit(tree)
                violations_by_rule["branchless"].extend(
                    _serialize_branchless(item) for item in branchless_visitor.violations
                )

                defensive_visitor = defensive_fallback_policy_check._DefensiveFallbackVisitor(
                    rel_path=rel_path,
                    source_lines=source_lines,
                )
                defensive_visitor.visit(tree)
                violations_by_rule["defensive_fallback"].extend(
                    _serialize_defensive(item) for item in defensive_visitor.violations
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
        }
    normalized: dict[str, list[dict[str, object]]] = {}
    for rule in ("no_monkeypatch", "branchless", "defensive_fallback"):
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
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _parse_tree(source: str, *, rel_path: str):
    import ast

    try:
        return ast.parse(source)
    except SyntaxError:
        # Surface syntax failures through existing rule scripts instead of reclassifying here.
        return None


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


__all__ = [
    "PolicySuiteResult",
    "load_or_scan_policy_suite",
    "scan_policy_suite",
    "violations_for_rule",
]
