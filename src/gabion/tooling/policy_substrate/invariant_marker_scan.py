from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import (
    DEFAULT_MARKER_ALIASES,
    MarkerKind,
    MarkerLifecycleState,
    MarkerPayload,
    SemanticLinkKind,
    SemanticReference,
    marker_identity,
    normalize_marker_payload,
)
from gabion.analysis.indexed_scan.scanners.marker_metadata import (
    keyword_links_literal,
    keyword_string_literal,
    never_reason,
)
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.site_identity import canonical_site_identity

_DEFAULT_SCAN_ROOTS = (
    Path("src") / "gabion",
    Path("src") / "gabion_governance",
    Path("scripts"),
)
_MARKER_FUNCTION_NAMES = {
    f"gabion.invariants.{alias}": kind
    for kind, aliases in DEFAULT_MARKER_ALIASES.items()
    for alias in aliases
    if "." not in alias
}
_MARKER_DECORATOR_NAMES = {
    "gabion.invariants.never_decorator": MarkerKind.NEVER,
    "gabion.invariants.todo_decorator": MarkerKind.TODO,
    "gabion.invariants.deprecated_decorator": MarkerKind.DEPRECATED,
}
_INVARIANT_DECORATOR_NAME = "gabion.invariants.invariant_decorator"
_MARKER_MEMBER_NAMES = frozenset(
    {
        "never",
        "todo",
        "deprecated",
        "never_decorator",
        "todo_decorator",
        "deprecated_decorator",
        "invariant_decorator",
    }
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.policy_substrate.invariant_marker_scan",
        key=key,
    )


def _check_deadline() -> None:
    return None


def _sort_once[T](
    values: Sequence[T],
    *,
    source: str,
    key=None,
) -> list[T]:
    return ordered_or_sorted(
        list(values),
        source=source,
        key=key,
    )


def _keyword_reasoning_literal(call: ast.Call) -> dict[str, object]:
    for keyword in call.keywords:
        if keyword.arg != "reasoning" or not isinstance(keyword.value, ast.Dict):
            continue
        payload: dict[str, object] = {}
        for raw_key, raw_value in zip(keyword.value.keys, keyword.value.values, strict=False):
            if raw_key is None:
                continue
            key = _const_text(raw_key)
            if not key:
                continue
            if key in {"summary", "control"}:
                value = _const_text(raw_value)
                if value:
                    payload[key] = value
                continue
            if key != "blocking_dependencies":
                continue
            values: list[str] = []
            if isinstance(raw_value, ast.List):
                values = [
                    value
                    for item in raw_value.elts
                    if (value := _const_text(item))
                ]
            else:
                scalar_value = _const_text(raw_value)
                if scalar_value:
                    values = [scalar_value]
            if values:
                payload[key] = _sort_once(
                    values,
                    source="tooling.policy_substrate.invariant_marker_scan.keyword_reasoning_literal",
                    key=str,
                )
        return payload
    return {}


@dataclass(frozen=True)
class InvariantMarkerScanNode:
    scan_kind: str
    marker_name: str
    marker_kind: str
    marker_id: str
    marker_reason: str
    owner: str
    expiry: str
    lifecycle_state: str
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    links: tuple[SemanticReference, ...]
    object_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    invariant_ids: tuple[str, ...]
    rel_path: str
    module: str
    qualname: str
    symbol: str
    line: int
    column: int
    ast_node_kind: str
    surface: str
    site_identity: str
    structural_identity: str
    valid: bool
    missing_fields: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "scan_kind": self.scan_kind,
            "marker_name": self.marker_name,
            "marker_kind": self.marker_kind,
            "marker_id": self.marker_id,
            "marker_reason": self.marker_reason,
            "owner": self.owner,
            "expiry": self.expiry,
            "lifecycle_state": self.lifecycle_state,
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "links": [
                {"kind": link.kind.value, "value": link.value}
                for link in self.links
            ],
            "object_ids": list(self.object_ids),
            "doc_ids": list(self.doc_ids),
            "policy_ids": list(self.policy_ids),
            "invariant_ids": list(self.invariant_ids),
            "rel_path": self.rel_path,
            "module": self.module,
            "qualname": self.qualname,
            "symbol": self.symbol,
            "line": self.line,
            "column": self.column,
            "ast_node_kind": self.ast_node_kind,
            "surface": self.surface,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "valid": self.valid,
            "missing_fields": list(self.missing_fields),
        }


def _scan_roots(root: Path) -> tuple[Path, ...]:
    return tuple(path for path in _DEFAULT_SCAN_ROOTS if (root / path).exists())


def _iter_python_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for scan_root in _scan_roots(root):
        files.extend(
            path
            for path in (root / scan_root).rglob("*.py")
            if path.is_file()
        )
    return tuple(_sorted(files))


def _module_name_for_path(root: Path, rel_path: str) -> str:
    path = Path(rel_path)
    if len(path.parts) >= 2 and path.parts[0] == "src":
        return ".".join(path.with_suffix("").parts[1:])
    if path.parts and path.parts[0] == "scripts":
        return ".".join(path.with_suffix("").parts)
    return ""


def _dotted_name(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=name):
            return name
        case ast.Attribute(value=value, attr=attr):
            parent = _dotted_name(value)
            if parent is None:
                return None
            return f"{parent}.{attr}"
        case _:
            return None


def _const_text(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value).strip()
    return ""


def _module_alias_maps(tree: ast.Module, *, module: str) -> tuple[dict[str, str], dict[str, str]]:
    module_aliases: dict[str, str] = {}
    direct_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "gabion.invariants":
                    module_aliases[alias.asname or "invariants"] = alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(
                module=module,
                level=node.level,
                imported_module=node.module,
            )
            for alias in node.names:
                if alias.name == "*":
                    continue
                full_name = ".".join(part for part in (base, alias.name) if part)
                if full_name == "gabion.invariants":
                    module_aliases[alias.asname or alias.name] = full_name
                elif full_name in _MARKER_FUNCTION_NAMES or full_name in _MARKER_DECORATOR_NAMES:
                    direct_aliases[alias.asname or alias.name] = full_name
                elif full_name == _INVARIANT_DECORATOR_NAME:
                    direct_aliases[alias.asname or alias.name] = full_name
    return (module_aliases, direct_aliases)


def _resolve_from_import(*, module: str, level: int, imported_module: str | None) -> str:
    if level == 0:
        return imported_module or ""
    base = module.split(".")[:-level]
    if imported_module:
        base += imported_module.split(".")
    return ".".join(base)


def _resolved_reference_name(
    node: ast.AST,
    *,
    module_aliases: Mapping[str, str],
    direct_aliases: Mapping[str, str],
) -> str:
    dotted = _dotted_name(node) or ""
    if not dotted:
        return ""
    if dotted in direct_aliases:
        return direct_aliases[dotted]
    head, _, tail = dotted.partition(".")
    if head in module_aliases:
        base = module_aliases[head]
        return f"{base}.{tail}" if tail else base
    return dotted


def _marker_kind_from_decorator_name(resolved_name: str, decorator: ast.Call) -> MarkerKind | None:
    if resolved_name in _MARKER_DECORATOR_NAMES:
        return _MARKER_DECORATOR_NAMES[resolved_name]
    if resolved_name != _INVARIANT_DECORATOR_NAME:
        return None
    marker_kind_value = ""
    if decorator.args:
        marker_kind_value = _const_text(decorator.args[0])
    for keyword in decorator.keywords:
        if keyword.arg == "marker_kind":
            marker_kind_value = _const_text(keyword.value) or marker_kind_value
    if marker_kind_value in {kind.value for kind in MarkerKind}:
        return MarkerKind(marker_kind_value)
    return None


def _lifecycle_state(call_or_decorator: ast.Call) -> MarkerLifecycleState:
    lifecycle_value = keyword_string_literal(
        call_or_decorator,
        "lifecycle_state",
        check_deadline_fn=_check_deadline,
    ).lower()
    if lifecycle_value == MarkerLifecycleState.EXPIRED.value:
        return MarkerLifecycleState.EXPIRED
    if lifecycle_value == MarkerLifecycleState.ROLLED_BACK.value:
        return MarkerLifecycleState.ROLLED_BACK
    return MarkerLifecycleState.ACTIVE


def _missing_fields(payload: MarkerPayload) -> tuple[str, ...]:
    missing: list[str] = []
    if not payload.reason.strip():
        missing.append("reason")
    if not payload.owner.strip():
        missing.append("owner")
    if not payload.expiry.strip():
        missing.append("expiry")
    if not payload.reasoning.summary.strip():
        missing.append("reasoning.summary")
    if not payload.reasoning.control.strip():
        missing.append("reasoning.control")
    if not payload.reasoning.blocking_dependencies:
        missing.append("reasoning.blocking_dependencies")
    if payload.lifecycle_state is not MarkerLifecycleState.ACTIVE:
        missing.append("lifecycle_state")
    return tuple(missing)


def _link_values(
    payload: MarkerPayload,
    *,
    kind: SemanticLinkKind,
) -> tuple[str, ...]:
    values = [
        link.value
        for link in payload.links
        if link.kind is kind and link.value.strip()
    ]
    return tuple(_sorted(values))


def _node_from_payload(
    *,
    scan_kind: str,
    marker_name: str,
    payload: MarkerPayload,
    rel_path: str,
    module: str,
    qualname: str,
    symbol: str,
    line: int,
    column: int,
    ast_node_kind: str,
    surface: str,
    structural_path: str,
) -> InvariantMarkerScanNode:
    site_identity = canonical_site_identity(
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        node_kind=ast_node_kind,
        surface=surface,
    )
    structural_identity = canonical_structural_identity(
        rel_path=rel_path,
        qualname=qualname,
        structural_path=structural_path,
        node_kind=ast_node_kind,
        surface=surface,
    )
    missing_fields = _missing_fields(payload)
    return InvariantMarkerScanNode(
        scan_kind=scan_kind,
        marker_name=marker_name,
        marker_kind=payload.marker_kind.value,
        marker_id=marker_identity(payload),
        marker_reason=payload.reason,
        owner=payload.owner,
        expiry=payload.expiry,
        lifecycle_state=payload.lifecycle_state.value,
        reasoning_summary=payload.reasoning.summary,
        reasoning_control=payload.reasoning.control,
        blocking_dependencies=payload.reasoning.blocking_dependencies,
        links=payload.links,
        object_ids=_link_values(payload, kind=SemanticLinkKind.OBJECT_ID),
        doc_ids=_link_values(payload, kind=SemanticLinkKind.DOC_ID),
        policy_ids=_link_values(payload, kind=SemanticLinkKind.POLICY_ID),
        invariant_ids=_link_values(payload, kind=SemanticLinkKind.INVARIANT_ID),
        rel_path=rel_path,
        module=module,
        qualname=qualname,
        symbol=symbol,
        line=line,
        column=column,
        ast_node_kind=ast_node_kind,
        surface=surface,
        site_identity=site_identity,
        structural_identity=structural_identity,
        valid=not missing_fields,
        missing_fields=missing_fields,
    )


class _InvariantMarkerVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        root: Path,
        rel_path: str,
        module: str,
        module_aliases: Mapping[str, str],
        direct_aliases: Mapping[str, str],
    ) -> None:
        self.root = root
        self.rel_path = rel_path
        self.module = module
        self.module_aliases = module_aliases
        self.direct_aliases = direct_aliases
        self._scope: list[str] = []
        self._callsite_ordinals: defaultdict[str, int] = defaultdict(int)
        self.nodes: list[InvariantMarkerScanNode] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_decorated_symbol(
            node=node,
            symbol=node.name,
            ast_node_kind="class_def",
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_decorated_symbol(
            node=node,
            symbol=node.name,
            ast_node_kind="function_def",
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_decorated_symbol(
            node=node,
            symbol=node.name,
            ast_node_kind="async_function_def",
        )
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        resolved_name = _resolved_reference_name(
            node.func,
            module_aliases=self.module_aliases,
            direct_aliases=self.direct_aliases,
        )
        marker_kind = _MARKER_FUNCTION_NAMES.get(resolved_name)
        if marker_kind is not None:
            reason = str(
                never_reason(node, check_deadline_fn=_check_deadline) or ""
            ).strip()
            payload = normalize_marker_payload(
                reason=reason,
                marker_kind=marker_kind,
                owner=keyword_string_literal(
                    node,
                    "owner",
                    check_deadline_fn=_check_deadline,
                ),
                expiry=keyword_string_literal(
                    node,
                    "expiry",
                    check_deadline_fn=_check_deadline,
                ),
                lifecycle_state=_lifecycle_state(node),
                links=tuple(
                    dict(item)
                    for item in keyword_links_literal(
                        node,
                        check_deadline_fn=_check_deadline,
                        sort_once_fn=_sort_once,
                    )
                ),
                reasoning=_keyword_reasoning_literal(node),
            )
            qualname = ".".join(self._scope) if self._scope else "<module>"
            scope_key = qualname or "<module>"
            self._callsite_ordinals[scope_key] += 1
            ordinal = self._callsite_ordinals[scope_key]
            self.nodes.append(
                _node_from_payload(
                    scan_kind="marker_callsite",
                    marker_name=resolved_name,
                    payload=payload,
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    symbol=resolved_name.rsplit(".", 1)[-1],
                    line=int(node.lineno),
                    column=int(node.col_offset) + 1,
                    ast_node_kind="call",
                    surface="marker_callsite",
                    structural_path=f"{scope_key}::marker_call[{ordinal}:{resolved_name}]",
                )
            )
        self.generic_visit(node)

    def _visit_decorated_symbol(
        self,
        *,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        symbol: str,
        ast_node_kind: str,
    ) -> None:
        qualname = ".".join([*self._scope, symbol])
        for ordinal, decorator in enumerate(node.decorator_list, start=1):
            if not isinstance(decorator, ast.Call):
                continue
            resolved_name = _resolved_reference_name(
                decorator.func,
                module_aliases=self.module_aliases,
                direct_aliases=self.direct_aliases,
            )
            marker_kind = _marker_kind_from_decorator_name(
                resolved_name,
                decorator,
            )
            if marker_kind is None:
                continue
            reason = keyword_string_literal(
                decorator,
                "reason",
                check_deadline_fn=_check_deadline,
            )
            if not reason and len(decorator.args) > 1:
                reason = _const_text(decorator.args[1])
            payload = normalize_marker_payload(
                reason=reason,
                marker_kind=marker_kind,
                owner=keyword_string_literal(
                    decorator,
                    "owner",
                    check_deadline_fn=_check_deadline,
                ),
                expiry=keyword_string_literal(
                    decorator,
                    "expiry",
                    check_deadline_fn=_check_deadline,
                ),
                lifecycle_state=_lifecycle_state(decorator),
                links=tuple(
                    dict(item)
                    for item in keyword_links_literal(
                        decorator,
                        check_deadline_fn=_check_deadline,
                        sort_once_fn=_sort_once,
                    )
                ),
                reasoning=_keyword_reasoning_literal(decorator),
            )
            self.nodes.append(
                _node_from_payload(
                    scan_kind="decorated_symbol",
                    marker_name=resolved_name,
                    payload=payload,
                    rel_path=self.rel_path,
                    module=self.module,
                    qualname=qualname,
                    symbol=symbol,
                    line=int(node.lineno),
                    column=int(node.col_offset) + 1,
                    ast_node_kind=ast_node_kind,
                    surface="decorated_symbol",
                    structural_path=f"{qualname}::decorator[{ordinal}:{payload.marker_kind.value}]",
                )
            )


def scan_invariant_markers(root: Path) -> tuple[InvariantMarkerScanNode, ...]:
    nodes: list[InvariantMarkerScanNode] = []
    for path in _iter_python_files(root):
        rel_path = path.relative_to(root).as_posix()
        module = _module_name_for_path(root, rel_path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel_path)
        module_aliases, direct_aliases = _module_alias_maps(tree, module=module)
        visitor = _InvariantMarkerVisitor(
            root=root,
            rel_path=rel_path,
            module=module,
            module_aliases=module_aliases,
            direct_aliases=direct_aliases,
        )
        visitor.visit(tree)
        nodes.extend(visitor.nodes)
    return tuple(
        _sorted(
            nodes,
            key=lambda item: (
                item.rel_path,
                item.line,
                item.column,
                item.qualname,
                item.marker_id,
            ),
        )
    )


def decorated_symbol_marker_index(
    nodes: Iterable[InvariantMarkerScanNode],
) -> dict[tuple[str, str, int], InvariantMarkerScanNode]:
    index: dict[tuple[str, str, int], InvariantMarkerScanNode] = {}
    for node in nodes:
        if node.scan_kind != "decorated_symbol":
            continue
        index[(node.rel_path, node.symbol, node.line)] = node
    return index


__all__ = [
    "InvariantMarkerScanNode",
    "decorated_symbol_marker_index",
    "scan_invariant_markers",
]
