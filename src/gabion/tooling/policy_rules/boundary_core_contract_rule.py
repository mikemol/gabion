#!/usr/bin/env python3
# gabion:decision_protocol_module
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TARGET_GLOB = "src/gabion/**/*.py"
BOUNDARY_MARKER = "gabion:boundary_normalization_module"


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    column: int
    qualname: str
    kind: str
    message: str
    structured_hash: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.kind}:{self.structured_hash}"

    @property
    def legacy_key(self) -> str:
        return f"{self.path}:{self.qualname}:{self.line}:{self.kind}"

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: [{self.qualname}] {self.message}"


@dataclass(frozen=True)
class _CoreImport:
    module_dotted: str
    alias_names: tuple[str, ...]


def collect_violations(*, root: Path, files: Sequence[Path] | None = None) -> list[Violation]:
    candidates = files if files is not None else tuple(sorted(root.glob(TARGET_GLOB)))
    violations: list[Violation] = []
    for path in candidates:
        path_is_scannable = path.is_file() and not any(
            part == "__pycache__" for part in path.parts
        )
        if path_is_scannable:
            rel_path = path.relative_to(root).as_posix()
            source = _read_source(path)
            if source is not None:
                source_lines = source.splitlines()
                if _module_has_boundary_marker(source_lines):
                    tree = _parse_tree(source)
                    if tree is None:
                        violations.append(
                            _violation(
                                rel_path=rel_path,
                                line=1,
                                column=1,
                                qualname="<module>",
                                kind="syntax_error",
                                message="unable to parse boundary module for boundary/core contract checks",
                            )
                        )
                    else:
                        core_imports = _collect_core_imports(tree=tree, rel_path=rel_path)
                        if not core_imports:
                            violations.append(
                                _violation(
                                    rel_path=rel_path,
                                    line=1,
                                    column=1,
                                    qualname="<module>",
                                    kind="missing_paired_core_module",
                                    message="boundary normalization module must import at least one paired *_core module",
                                )
                            )
                        else:
                            if not _has_explicit_single_hop_core_call(
                                tree=tree,
                                core_imports=core_imports,
                            ):
                                violations.append(
                                    _violation(
                                        rel_path=rel_path,
                                        line=1,
                                        column=1,
                                        qualname="<module>",
                                        kind="missing_single_hop_core_call",
                                        message=(
                                            "boundary module must call paired core via "
                                            "explicit single-hop boundary->core call"
                                        ),
                                    )
                                )

                            for core_import in core_imports:
                                core_path = _module_path_from_dotted(
                                    root=root,
                                    dotted=core_import.module_dotted,
                                )
                                if core_path is None or not core_path.exists():
                                    violations.append(
                                        _violation(
                                            rel_path=rel_path,
                                            line=1,
                                            column=1,
                                            qualname="<module>",
                                            kind="missing_core_module_file",
                                            message=(
                                                "paired core module "
                                                f"'{core_import.module_dotted}' must resolve "
                                                "to an on-disk module"
                                            ),
                                        )
                                    )
                                else:
                                    core_source = _read_source(core_path)
                                    if core_source is not None:
                                        core_rel = core_path.relative_to(root).as_posix()
                                        core_lines = core_source.splitlines()
                                        core_tree = _parse_tree(core_source)
                                        if core_tree is None:
                                            violations.append(
                                                _violation(
                                                    rel_path=core_rel,
                                                    line=1,
                                                    column=1,
                                                    qualname="<module>",
                                                    kind="core_syntax_error",
                                                    message=(
                                                        "paired core module must parse cleanly "
                                                        "for contract checks"
                                                    ),
                                                )
                                            )
                                        else:
                                            if _module_has_boundary_marker(core_lines):
                                                violations.append(
                                                    _violation(
                                                        rel_path=core_rel,
                                                        line=1,
                                                        column=1,
                                                        qualname="<module>",
                                                        kind="core_marked_as_boundary",
                                                        message=(
                                                            "paired core module must not be marked as "
                                                            "boundary_normalization_module"
                                                        ),
                                                    )
                                                )

                                            violations.extend(
                                                _core_annotation_violations(
                                                    rel_path=core_rel,
                                                    tree=core_tree,
                                                )
                                            )
                                            violations.extend(
                                                _core_narrowing_violations(
                                                    rel_path=core_rel,
                                                    tree=core_tree,
                                                )
                                            )
                                            violations.extend(
                                                _core_branch_violations(
                                                    rel_path=core_rel,
                                                    tree=core_tree,
                                                )
                                            )

    return _dedupe_exact_violations(violations)


def _read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_tree(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _module_has_boundary_marker(source_lines: list[str]) -> bool:
    for raw in source_lines[:100]:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and BOUNDARY_MARKER in stripped:
            return True
    return False


def _collect_core_imports(*, tree: ast.AST, rel_path: str) -> tuple[_CoreImport, ...]:
    imports: list[_CoreImport] = []
    package_parts = _package_parts_from_rel_path(rel_path)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=import_names):
                for alias in import_names:
                    dotted = alias.name.strip()
                    if dotted.endswith("_core"):
                        alias_name = (alias.asname or dotted.rsplit(".", 1)[-1]).strip()
                        imports.append(
                            _CoreImport(module_dotted=dotted, alias_names=(alias_name,))
                        )
            case ast.ImportFrom(module=module, level=level, names=import_names):
                module_name = (module or "").strip()
                resolved_module = _resolve_import_from_module(
                    module=module_name,
                    level=int(level or 0),
                    package_parts=package_parts,
                )
                if resolved_module is not None:
                    if resolved_module.endswith("_core"):
                        alias_names = tuple(
                            (alias.asname or alias.name).strip()
                            for alias in import_names
                            if alias.name != "*"
                        )
                        if alias_names:
                            imports.append(
                                _CoreImport(
                                    module_dotted=resolved_module,
                                    alias_names=alias_names,
                                )
                            )
                    else:
                        for alias in import_names:
                            imported_name = alias.name.strip()
                            if imported_name != "*" and imported_name.endswith("_core"):
                                imports.append(
                                    _CoreImport(
                                        module_dotted=f"{resolved_module}.{imported_name}",
                                        alias_names=((alias.asname or imported_name).strip(),),
                                    )
                                )
            case _:
                pass
    deduped: dict[str, set[str]] = {}
    for item in imports:
        entry = deduped.setdefault(item.module_dotted, set())
        entry.update(name for name in item.alias_names if name)
    return tuple(
        _CoreImport(module_dotted=module, alias_names=tuple(sorted(names)))
        for module, names in sorted(deduped.items())
    )


def _package_parts_from_rel_path(rel_path: str) -> tuple[str, ...]:
    path = Path(rel_path)
    parts = list(path.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        return tuple(parts)
    return tuple(parts[:-1])


def _resolve_import_from_module(
    *,
    module: str,
    level: int,
    package_parts: tuple[str, ...],
) -> str | None:
    if level <= 0:
        return module or None
    if not package_parts:
        return None
    if level == 1:
        base = package_parts
    else:
        trim = level - 1
        if trim > len(package_parts):
            return None
        base = package_parts[: len(package_parts) - trim]
    base_text = ".".join(base)
    if module:
        if base_text:
            return f"{base_text}.{module}"
        return module
    return base_text or None


def _module_path_from_dotted(*, root: Path, dotted: str) -> Path | None:
    if not dotted.startswith("gabion."):
        return None
    rel = Path("src") / Path(*dotted.split("."))
    return root / rel.with_suffix(".py")


def _has_explicit_single_hop_core_call(*, tree: ast.AST, core_imports: Sequence[_CoreImport]) -> bool:
    module_aliases = {name for item in core_imports for name in item.alias_names}
    for node in ast.walk(tree):
        match node:
            case ast.Call(func=ast.Attribute(value=ast.Name(id=module_alias_id))):
                if module_alias_id in module_aliases:
                    return True
            case ast.Call(func=ast.Name(id=callable_name)):
                if callable_name in module_aliases:
                    return True
            case _:
                pass
    return False


def _core_annotation_violations(*, rel_path: str, tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                annotations: list[tuple[int, int, str]] = []
                for arg in [*node.args.args, *node.args.kwonlyargs]:
                    if arg.annotation is not None:
                        annotations.append(
                            (
                                int(arg.lineno or node.lineno),
                                int(arg.col_offset or node.col_offset) + 1,
                                ast.unparse(arg.annotation),
                            )
                        )
                if node.args.vararg and node.args.vararg.annotation is not None:
                    arg = node.args.vararg
                    annotations.append(
                        (
                            int(arg.lineno or node.lineno),
                            int(arg.col_offset or node.col_offset) + 1,
                            ast.unparse(arg.annotation),
                        )
                    )
                if node.args.kwarg and node.args.kwarg.annotation is not None:
                    arg = node.args.kwarg
                    annotations.append(
                        (
                            int(arg.lineno or node.lineno),
                            int(arg.col_offset or node.col_offset) + 1,
                            ast.unparse(arg.annotation),
                        )
                    )
                if node.returns is not None:
                    annotations.append(
                        (
                            int(node.returns.lineno or node.lineno),
                            int(node.returns.col_offset or node.col_offset) + 1,
                            ast.unparse(node.returns),
                        )
                    )

                for line, col, text in annotations:
                    compact = text.replace(" ", "")
                    if (
                        text == "Any"
                        or text == "object"
                        or compact == "dict[str,object]"
                    ):
                        violations.append(
                            _violation(
                                rel_path=rel_path,
                                line=line,
                                column=col,
                                qualname=node.name,
                                kind="raw_ingress_type_in_core",
                                message=(
                                    "core signature must not expose raw ingress types "
                                    "(Any/object/dict[str, object])"
                                ),
                            )
                        )
            case _:
                pass
    return violations


def _core_narrowing_violations(*, rel_path: str, tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        match node:
            case ast.Call(func=ast.Name(id="isinstance")) | ast.Call(
                func=ast.Name(id="cast")
            ):
                violations.append(
                    _violation(
                        rel_path=rel_path,
                        line=int(getattr(node, "lineno", 1) or 1),
                        column=int(getattr(node, "col_offset", 0) or 0) + 1,
                        qualname="<module>",
                        kind="ingress_narrowing_in_core",
                        message="core module must not perform runtime ingress narrowing",
                    )
                )
            case _:
                pass
    return violations


def _core_branch_violations(*, rel_path: str, tree: ast.AST) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        match node:
            case (
                ast.If()
                | ast.IfExp()
                | ast.Match()
                | ast.For()
                | ast.AsyncFor()
                | ast.While()
                | ast.Try()
                | ast.TryStar()
            ):
                violations.append(
                    _violation(
                        rel_path=rel_path,
                        line=int(getattr(node, "lineno", 1) or 1),
                        column=int(getattr(node, "col_offset", 0) or 0) + 1,
                        qualname="<module>",
                        kind="branch_in_core_module",
                        message="paired core module must remain branchless",
                    )
                )
            case _:
                pass
    return violations


def _violation(
    *,
    rel_path: str,
    line: int,
    column: int,
    qualname: str,
    kind: str,
    message: str,
) -> Violation:
    structured_hash = _structured_hash(rel_path, qualname, kind, str(column), message)
    return Violation(
        path=rel_path,
        line=line,
        column=column,
        qualname=qualname,
        kind=kind,
        message=message,
        structured_hash=structured_hash,
    )


def _dedupe_exact_violations(violations: Sequence[Violation]) -> list[Violation]:
    deduped: list[Violation] = []
    seen: set[tuple[str, int, int, str, str, str]] = set()
    for violation in violations:
        signature = (
            violation.path,
            violation.line,
            violation.column,
            violation.qualname,
            violation.kind,
            violation.message,
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(violation)
    return deduped


def _structured_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = [
    "Violation",
    "collect_violations",
]
