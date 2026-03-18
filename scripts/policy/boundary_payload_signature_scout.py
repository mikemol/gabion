#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import TypeGuard

BOUNDARY_MARKER = "gabion:boundary_normalization_module"
TARGET_GLOB = "src/gabion/**/*.py"

FORBIDDEN_FIRST_PARAM_ANNOTATIONS = {
    "dict[str,object]",
    "Mapping[str,object]",
    "MutableMapping[str,object]",
    "collections.abc.Mapping[str,object]",
    "collections.abc.MutableMapping[str,object]",
}


@dataclass(frozen=True)
class ScoutFinding:
    path: str
    line: int
    function: str
    first_param: str
    return_annotation: str
    suggestion: str | None

    def render(self) -> str:
        suggestion_text = self.suggestion if self.suggestion is not None else "-"
        return (
            f"{self.path}:{self.line}: [{self.function}] first={self.first_param} "
            f"return={self.return_annotation or '-'} suggestion={suggestion_text}"
        )


def _normalize_annotation_text(text: str) -> str:
    return "".join(text.split())


def _module_has_boundary_marker(source: str) -> bool:
    for raw_line in source.splitlines()[:80]:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and BOUNDARY_MARKER in stripped:
            return True
        if stripped.startswith("\"\"\"") or stripped.startswith("'''"):
            continue
    return False


def _suggested_annotation(first_param: str, return_annotation: str) -> str | None:
    if return_annotation in {
        "dict[str,JSONObject]",
        "Mapping[str,JSONObject]",
        "collections.abc.Mapping[str,JSONObject]",
    }:
        return "Mapping[str, JSONObject]"
    if return_annotation in {
        "dict[str,JSONValue]",
        "Mapping[str,JSONValue]",
        "collections.abc.Mapping[str,JSONValue]",
    }:
        return "Mapping[str, JSONValue]"
    if return_annotation == "dict[str,str]":
        return "Mapping[str, str]"
    if first_param.startswith("dict["):
        return "Mapping[str, object]"
    return None


def _first_param_annotation(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, str]:
    params = [*function_node.args.posonlyargs, *function_node.args.args]
    first = params[0]
    first_text = _normalize_annotation_text(ast.unparse(first.annotation))
    line = int(getattr(first, "lineno", getattr(function_node, "lineno", 1)) or 1)
    return line, first_text


def _has_annotated_first_param(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    params = [*function_node.args.posonlyargs, *function_node.args.args]
    return bool(params and params[0].annotation is not None)


def _is_private_function_node(
    node: ast.AST,
) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_")


def _is_scout_candidate_path(path: Path) -> bool:
    return path.is_file() and "__pycache__" not in path.parts


def _scout_candidate_paths(root: Path) -> Iterator[Path]:
    return filter(_is_scout_candidate_path, sorted(root.glob(TARGET_GLOB)))


def _annotated_private_function_nodes(
    module: ast.AST,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int, str]]:
    for node in filter(
        _has_annotated_first_param,
        filter(_is_private_function_node, ast.walk(module)),
    ):
        line, first_param = _first_param_annotation(node)
        yield node, line, first_param


def _iter_findings_for_path(path: Path, *, root: Path) -> Iterator[ScoutFinding]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return iter(())
    if not _module_has_boundary_marker(source):
        return iter(())
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError:
        return iter(())
    rel = path.relative_to(root).as_posix()
    return (
        ScoutFinding(
            path=rel,
            line=line,
            function=node.name,
            first_param=first_param,
            return_annotation=(
                _normalize_annotation_text(ast.unparse(node.returns))
                if node.returns is not None
                else ""
            ),
            suggestion=_suggested_annotation(first_param, return_annotation),
        )
        for node, line, first_param in _annotated_private_function_nodes(module)
        if first_param in FORBIDDEN_FIRST_PARAM_ANNOTATIONS
        for return_annotation in (
            _normalize_annotation_text(ast.unparse(node.returns))
            if node.returns is not None
            else "",
        )
    )


def collect_findings(*, root: Path) -> list[ScoutFinding]:
    return list(
        chain.from_iterable(
            _iter_findings_for_path(path, root=root)
            for path in _scout_candidate_paths(root)
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scout boundary modules for helper first-parameter annotations that will "
            "trip non-boundary payload signature policy after marker removal."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    findings = collect_findings(root=root)

    print(
        "boundary-payload-signature-scout: "
        f"findings={len(findings)} root={root.as_posix()}"
    )
    for finding in findings:
        print(f"  - {finding.render()}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": root.as_posix(),
            "finding_count": len(findings),
            "findings": [asdict(finding) for finding in findings],
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"boundary-payload-signature-scout: wrote {args.out.as_posix()}")

    if args.fail_on_findings and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
