#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


TARGET_GLOB = "src/gabion/**/*.py"
Category = Literal[
    "ingress_normalization",
    "core_preservation",
    "egress_enforcement",
    "forbidden",
]

_INGRESS_HINTS = (
    "ingress",
    "normalize",
    "payload_in",
    "decode",
    "load",
    "parse",
    "request",
)
_EGRESS_HINTS = (
    "egress",
    "emit",
    "write",
    "dump",
    "report",
    "output",
    "response",
    "serialize",
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.message}"


@dataclass(frozen=True)
class CallsiteRecord:
    path: str
    line: int
    column: int
    kind: str
    category: Category
    source: str | None = None


@dataclass
class FileInventory:
    path: str
    sorted_calls: int = 0
    dot_sort_calls: int = 0
    ordered_or_sorted_calls: int = 0
    sort_once_calls: int = 0
    json_dumps_sort_keys_true_calls: int = 0
    callsites: list[CallsiteRecord] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sorted_calls": self.sorted_calls,
            "dot_sort_calls": self.dot_sort_calls,
            "ordered_or_sorted_calls": self.ordered_or_sorted_calls,
            "sort_once_calls": self.sort_once_calls,
            "json_dumps_sort_keys_true_calls": self.json_dumps_sort_keys_true_calls,
            "callsites": [asdict(record) for record in self.callsites],
        }


class _OrderLifetimeVisitor(ast.NodeVisitor):
    def __init__(self, *, rel_path: str) -> None:
        self.rel_path = rel_path
        self.violations: list[Violation] = []
        self.inventory = FileInventory(path=rel_path)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _call_name(node.func)

        if callee == "sorted":
            self.inventory.sorted_calls += 1
            self._record_callsite(node, kind="sorted", category="forbidden")
            self._report(
                node,
                "raw sorted(...) forbidden by single-sort lifetime ratchet; use sort_once(...)",
            )

        if isinstance(node.func, ast.Attribute) and node.func.attr == "sort":
            self.inventory.dot_sort_calls += 1
            self._record_callsite(node, kind="dot_sort", category="forbidden")
            self._report(
                node,
                "raw .sort(...) forbidden by single-sort lifetime ratchet; use sort_once(...)",
            )

        if callee == "json.dumps":
            sort_keys_kw = _keyword_value(node, "sort_keys")
            if _is_true_literal(sort_keys_kw):
                self.inventory.json_dumps_sort_keys_true_calls += 1
                self._record_callsite(node, kind="json.dumps.sort_keys_true", category="forbidden")
                self._report(
                    node,
                    "json.dumps(sort_keys=True) forbidden by single-sort lifetime ratchet",
                )

        if callee.endswith(".ordered_or_sorted") or callee == "ordered_or_sorted":
            self.inventory.ordered_or_sorted_calls += 1
            category = (
                "forbidden"
                if _is_active_ordered_or_sorted_call(node)
                else "core_preservation"
            )
            self._record_callsite(node, kind="ordered_or_sorted", category=category)
            if category == "forbidden":
                self._report(
                    node,
                    "direct active-sort mode via ordered_or_sorted(...) forbidden; use sort_once(...)",
                )

        if callee.endswith(".sort_once") or callee == "sort_once":
            self.inventory.sort_once_calls += 1
            source_kw = _keyword_value(node, "source")
            source_text = _source_text(source_kw)
            if source_kw is None:
                category = "forbidden"
                self._record_callsite(
                    node,
                    kind="sort_once",
                    category=category,
                    source=source_text,
                )
                self._report(
                    node,
                    "sort_once(...) requires source=... metadata",
                )
                self.generic_visit(node)
                return
            if source_text is None:
                category = "forbidden"
                self._record_callsite(
                    node,
                    kind="sort_once",
                    category=category,
                    source=source_text,
                )
                self._report(
                    node,
                    "sort_once(source=...) must be string literal or f-string",
                )
                self.generic_visit(node)
                return
            category = _classify_sort_once_category(source_text)
            self._record_callsite(
                node,
                kind="sort_once",
                category=category,
                source=source_text,
            )

        self.generic_visit(node)

    def _record_callsite(
        self,
        node: ast.Call,
        *,
        kind: str,
        category: Category,
        source: str | None = None,
    ) -> None:
        self.inventory.callsites.append(
            CallsiteRecord(
                path=self.rel_path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                kind=kind,
                category=category,
                source=source,
            )
        )

    def _report(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            Violation(
                path=self.rel_path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                message=message,
            )
        )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _keyword_value(node: ast.Call, name: str) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _is_true_literal(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def _is_false_literal(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is False


def _source_text(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    if isinstance(value, ast.JoinedStr):
        return "<fstring>"
    return None


def _classify_sort_once_category(source: str | None) -> Category:
    if not source:
        return "core_preservation"
    lowered = source.lower()
    if any(token in lowered for token in _INGRESS_HINTS):
        return "ingress_normalization"
    if any(token in lowered for token in _EGRESS_HINTS):
        return "egress_enforcement"
    return "core_preservation"


def _policy_is_sort(value: ast.expr | None) -> bool:
    if value is None:
        return False
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value.strip().lower() == "sort"
    if isinstance(value, ast.Attribute):
        return value.attr.upper() == "SORT"
    return False


def _is_active_ordered_or_sorted_call(node: ast.Call) -> bool:
    policy_kw = _keyword_value(node, "policy")
    require_sorted_kw = _keyword_value(node, "require_sorted")
    if _is_false_literal(require_sorted_kw):
        return True
    if require_sorted_kw is None and policy_kw is None:
        return True
    if _policy_is_sort(policy_kw):
        return True
    return False


def collect_violations_and_inventory(
    *,
    root: Path,
) -> tuple[list[Violation], list[FileInventory]]:
    violations: list[Violation] = []
    inventories: list[FileInventory] = []
    for path in sorted(root.glob(TARGET_GLOB)):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            violations.append(
                Violation(
                    path=rel_path,
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    message="syntax error while parsing module for order-lifetime checks",
                )
            )
            continue
        visitor = _OrderLifetimeVisitor(rel_path=rel_path)
        visitor.visit(tree)
        inventories.append(visitor.inventory)
        violations.extend(visitor.violations)
    return violations, inventories


def collect_violations(*, root: Path) -> list[Violation]:
    violations, _inventories = collect_violations_and_inventory(root=root)
    return violations


def _totals(inventories: list[FileInventory]) -> dict[str, int]:
    return {
        "sorted_calls": sum(item.sorted_calls for item in inventories),
        "dot_sort_calls": sum(item.dot_sort_calls for item in inventories),
        "ordered_or_sorted_calls": sum(item.ordered_or_sorted_calls for item in inventories),
        "sort_once_calls": sum(item.sort_once_calls for item in inventories),
        "json_dumps_sort_keys_true_calls": sum(
            item.json_dumps_sort_keys_true_calls for item in inventories
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce single-sort lifetime ratchet on src/gabion surfaces.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root (default: current directory)",
    )
    parser.add_argument(
        "--emit-inventory",
        type=Path,
        default=None,
        help="Write JSON inventory of ordering surfaces to this path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    violations, inventories = collect_violations_and_inventory(root=root)
    if args.emit_inventory is not None:
        payload = {
            "target_glob": TARGET_GLOB,
            "totals": _totals(inventories),
            "files": [item.as_dict() for item in sorted(inventories, key=lambda x: x.path)],
        }
        inventory_path = args.emit_inventory
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    if not violations:
        print("order-lifetime-check: no violations detected")
        return 0
    for violation in violations:
        print(violation.render())
    print(f"order-lifetime-check: {len(violations)} violation(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
