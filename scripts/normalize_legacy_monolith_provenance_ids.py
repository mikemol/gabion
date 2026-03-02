"""Normalize stale runtime provenance strings to stable symbolic IDs.

This codemod targets string literals bound to `source=` or `scope=` keyword
arguments in calls under src/gabion/analysis.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


NUMERIC_RUNTIME_PATH = re.compile(r"^src/gabion/analysis/legacy_dataflow_monolith\.py:(\d+)$")
SYMBOLIC_RUNTIME_PATH = re.compile(
    r"^src/gabion/analysis/legacy_dataflow_monolith\.py:(_[A-Za-z0-9_.]+)$"
)
SYMBOLIC_RUNTIME_SCOPE = re.compile(r"^legacy_dataflow_monolith\.(_[A-Za-z0-9_.]+)$")


@dataclass(frozen=True)
class Replacement:
    file_path: Path
    start: int
    end: int
    line: int
    col: int
    keyword: str
    owner: str
    rule: str
    old_value: str
    new_value: str
    old_literal: str
    new_literal: str


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, ch in enumerate(text):
        if ch == "\n":
            offsets.append(index + 1)
    return offsets


def _to_offset(offsets: list[int], lineno: int, col: int) -> int:
    return offsets[lineno - 1] + col


def _format_literal(original_literal: str, value: str) -> str:
    # Preserve quote style when possible; use JSON quoting as fallback.
    if len(original_literal) >= 2 and original_literal[0] == original_literal[-1]:
        quote = original_literal[0]
        if quote in {"'", '"'}:
            escaped = (
                value.replace("\\", "\\\\")
                .replace(quote, f"\\{quote}")
                .replace("\n", "\\n")
            )
            return f"{quote}{escaped}{quote}"
    return json.dumps(value)


def _nearest_owner(stack: list[tuple[str, str]], module_name: str) -> str:
    for index in range(len(stack) - 1, -1, -1):
        kind, name = stack[index]
        if kind != "function":
            continue
        owner = name
        if index > 0 and stack[index - 1][0] == "class":
            owner = f"{stack[index - 1][1]}.{name}"
        return owner
    return module_name


class _Collector(ast.NodeVisitor):
    def __init__(self, file_path: Path, source_text: str, module_name: str) -> None:
        self.file_path = file_path
        self.source_text = source_text
        self.module_name = module_name
        self.offsets = _line_offsets(source_text)
        self.stack: list[tuple[str, str]] = []
        self.per_owner_counts: dict[str, int] = {}
        self.replacements: list[Replacement] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.stack.append(("class", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.stack.append(("function", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.stack.append(("function", node.name))
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        for keyword in node.keywords:
            if keyword.arg not in {"source", "scope"}:
                continue
            if not isinstance(keyword.value, ast.Constant) or not isinstance(
                keyword.value.value, str
            ):
                continue
            start = _to_offset(
                self.offsets, keyword.value.lineno, keyword.value.col_offset
            )
            end = _to_offset(
                self.offsets, keyword.value.end_lineno, keyword.value.end_col_offset
            )
            old_literal = self.source_text[start:end]
            old_value = keyword.value.value
            owner = _nearest_owner(self.stack, self.module_name)
            rewrite = self._rewrite_value(keyword.arg, old_value, owner)
            if rewrite is None:
                continue
            rule, new_value = rewrite
            new_literal = _format_literal(old_literal, new_value)
            self.replacements.append(
                Replacement(
                    file_path=self.file_path,
                    start=start,
                    end=end,
                    line=keyword.value.lineno,
                    col=keyword.value.col_offset + 1,
                    keyword=keyword.arg,
                    owner=owner,
                    rule=rule,
                    old_value=old_value,
                    new_value=new_value,
                    old_literal=old_literal,
                    new_literal=new_literal,
                )
            )
        self.generic_visit(node)

    def _rewrite_value(self, keyword: str, old_value: str, owner: str) -> tuple[str, str] | None:
        if keyword == "scope" and old_value == "legacy_dataflow_monolith":
            return "scope_runtime_namespace", f"gabion.analysis.{self.module_name}"

        numeric_match = NUMERIC_RUNTIME_PATH.match(old_value)
        if numeric_match:
            count = self.per_owner_counts.get(owner, 0) + 1
            self.per_owner_counts[owner] = count
            return (
                "runtime_path_line_to_symbolic_id",
                f"gabion.analysis.{self.module_name}.{owner}.site_{count}",
            )

        symbolic_path_match = SYMBOLIC_RUNTIME_PATH.match(old_value)
        if symbolic_path_match:
            return (
                "runtime_path_symbol_to_owner_symbol",
                f"gabion.analysis.{self.module_name}.{symbolic_path_match.group(1)}",
            )

        symbolic_scope_match = SYMBOLIC_RUNTIME_SCOPE.match(old_value)
        if symbolic_scope_match:
            return (
                "runtime_scope_symbol_to_owner_symbol",
                f"gabion.analysis.{self.module_name}.{symbolic_scope_match.group(1)}",
            )
        return None


def _collect_replacements(file_path: Path, source_text: str) -> list[Replacement]:
    module_name = file_path.stem
    tree = ast.parse(source_text, filename=str(file_path))
    collector = _Collector(
        file_path=file_path, source_text=source_text, module_name=module_name
    )
    collector.visit(tree)
    return collector.replacements


def _apply_replacements(source_text: str, replacements: list[Replacement]) -> str:
    updated = source_text
    for repl in sorted(replacements, key=lambda item: item.start, reverse=True):
        updated = f"{updated[:repl.start]}{repl.new_literal}{updated[repl.end:]}"
    return updated


def _target_files(root: Path, analysis_root: Path) -> list[Path]:
    resolved_root = (root / analysis_root).resolve()
    return sorted(path for path in resolved_root.rglob("*.py") if path.is_file())


def _write_mapping(
    mapping_out: Path,
    replacements: list[Replacement],
    changed_files: list[str],
    root: Path,
) -> None:
    data = {
        "format_version": 1,
        "analysis_root": "src/gabion/analysis",
        "changed_files": changed_files,
        "replacement_count": len(replacements),
        "replacements": [
            {
                "file": str(item.file_path.relative_to(root)),
                "line": item.line,
                "col": item.col,
                "keyword": item.keyword,
                "owner": item.owner,
                "rule": item.rule,
                "old_value": item.old_value,
                "new_value": item.new_value,
            }
            for item in sorted(
                replacements, key=lambda entry: (str(entry.file_path), entry.start)
            )
        ],
    }
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text(f"{json.dumps(data, indent=2, sort_keys=True)}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--analysis-root",
        default="src/gabion/analysis",
        help="Path under --root containing analysis modules to rewrite.",
    )
    parser.add_argument(
        "--mapping-out",
        default="artifacts/audit_reports/runtime_provenance_id_map.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    analysis_root = Path(args.analysis_root)
    mapping_out = (root / args.mapping_out).resolve()

    all_replacements: list[Replacement] = []
    changed_files: list[str] = []
    for file_path in _target_files(root, analysis_root):
        source_text = file_path.read_text()
        replacements = _collect_replacements(file_path, source_text)
        if not replacements:
            continue
        updated = _apply_replacements(source_text, replacements)
        if updated != source_text:
            file_path.write_text(updated)
            changed_files.append(str(file_path.relative_to(root)))
        all_replacements.extend(replacements)

    _write_mapping(
        mapping_out=mapping_out,
        replacements=all_replacements,
        changed_files=sorted(changed_files),
        root=root,
    )
    print(
        "normalized runtime provenance literals:",
        len(all_replacements),
        "in",
        len(changed_files),
        "file(s)",
    )
    print("wrote mapping artifact:", mapping_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
