from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from gabion.analysis.timeout_context import check_deadline


_DOC_ROLE_RE = re.compile(r"^test_")


def _normalize_path(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<annotation>"


def _subscript_args(node: ast.Subscript) -> list[ast.AST]:
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_str(node: ast.AST) -> bool:
    return _name(node) == "str"


def _is_object_like(node: ast.AST) -> bool:
    return _name(node) in {"object", "Any"}


def _is_anonymous_dict_subscript(node: ast.Subscript) -> bool:
    container = _name(node.value)
    if container not in {"dict", "Dict", "Mapping", "MutableMapping"}:
        return False
    args = _subscript_args(node)
    if len(args) != 2:
        return False
    return _is_str(args[0]) and _is_object_like(args[1])


def _contains_anonymous_dict(annotation: ast.AST) -> bool:
    check_deadline()
    for node in ast.walk(annotation):
        check_deadline()
        if isinstance(node, ast.Subscript) and _is_anonymous_dict_subscript(node):
            return True
    return False


def _singularize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 2:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token


def _suggest_type_name(name: str) -> str | None:
    if not name:
        return None
    parts = [part for part in name.split("_") if part]
    if not parts:
        return None
    parts[-1] = _singularize_token(parts[-1])
    if "provenance" in parts and parts[-1] == "provenance":
        parts.append("entry")
    return "".join(part.capitalize() for part in parts)


@dataclass(frozen=True)
class AnonymousSchemaSurface:
    path: str
    lineno: int
    col: int
    context: str
    annotation: str
    suggestion: str | None = None

    def format(self) -> str:
        suffix = f" -> consider {self.suggestion}" if self.suggestion else ""
        return (
            f"{self.path}:{self.lineno}:{self.col}: {self.context} annotated as "
            f"{self.annotation}{suffix}"
        )


class _SurfaceVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []
        self.surfaces: list[AnonymousSchemaSurface] = []

    def _context_prefix(self) -> str:
        parts: list[str] = []
        if self._class_stack:
            parts.append(self._class_stack[-1])
        if self._func_stack:
            parts.append(self._func_stack[-1])
        return ".".join(parts)

    def _record(self, node: ast.AST, *, context: str, annotation: ast.AST, name_hint: str) -> None:
        suggestion = _suggest_type_name(name_hint)
        lineno = int(getattr(node, "lineno", 0) or 0)
        col = int(getattr(node, "col_offset", 0) or 0)
        self.surfaces.append(
            AnonymousSchemaSurface(
                path=str(self._path),
                lineno=lineno,
                col=col,
                context=context,
                annotation=_unparse(annotation),
                suggestion=suggestion,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node.name)
        if node.returns and _contains_anonymous_dict(node.returns):
            prefix = self._context_prefix()
            context = f"{prefix}.returns" if prefix else "returns"
            self._record(node, context=context, annotation=node.returns, name_hint=node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        if node.returns and _contains_anonymous_dict(node.returns):
            prefix = self._context_prefix()
            context = f"{prefix}.returns" if prefix else "returns"
            self._record(node, context=context, annotation=node.returns, name_hint=node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_arg(self, node: ast.arg) -> None:
        if node.annotation and _contains_anonymous_dict(node.annotation):
            prefix = self._context_prefix()
            context = f"{prefix}({node.arg})" if prefix else node.arg
            self._record(node, context=context, annotation=node.annotation, name_hint=node.arg)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.annotation and _contains_anonymous_dict(node.annotation):
            target = node.target
            name_hint = ""
            label = ""
            if isinstance(target, ast.Name):
                name_hint = target.id
                label = target.id
            else:
                label = _unparse(target)
            prefix = self._context_prefix()
            context = f"{prefix}.{label}" if prefix else label
            self._record(
                node,
                context=context,
                annotation=node.annotation,
                # Only suggest names for identifier-like targets; expressions like
                # `self.payload` are better left suggestion-free.
                name_hint=name_hint,
            )
        self.generic_visit(node)


def find_anonymous_schema_surfaces(
    paths: Iterable[Path],
    *,
    project_root: Path | None = None,
) -> list[AnonymousSchemaSurface]:
    """Find uses of dict[str, object] (and containers thereof) in annotations.

    These annotations typically indicate anonymous record/payload types whose
    schema is communicated by convention rather than by a first-class type.
    """
    check_deadline()
    surfaces: list[AnonymousSchemaSurface] = []
    for path in sorted(set(paths)):
        check_deadline()
        if "tests" in path.parts:
            continue
        if path.name.startswith("."):
            continue
        if _DOC_ROLE_RE.match(path.name):
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        visitor = _SurfaceVisitor(path)
        visitor.visit(tree)
        surfaces.extend(visitor.surfaces)
    normalized = [
        AnonymousSchemaSurface(
            path=_normalize_path(Path(surface.path), project_root),
            lineno=surface.lineno,
            col=surface.col,
            context=surface.context,
            annotation=surface.annotation,
            suggestion=surface.suggestion,
        )
        for surface in surfaces
    ]
    return sorted(
        normalized,
        key=lambda entry: (entry.path, entry.lineno, entry.col, entry.context),
    )
