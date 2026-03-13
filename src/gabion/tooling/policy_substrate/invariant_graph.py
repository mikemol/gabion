from __future__ import annotations

import ast
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from gabion.analysis.aspf.aspf_lattice_algebra import canonical_structural_identity
from gabion.analysis.foundation.marker_protocol import SemanticLinkKind
from gabion.order_contract import ordered_or_sorted
from gabion.tooling.policy_substrate.invariant_marker_scan import (
    InvariantMarkerScanNode,
    scan_invariant_markers,
)
from gabion.tooling.policy_substrate.projection_semantic_fragment_phase5_registry import (
    ProjectionSemanticFragmentPhase5QueueDefinition,
    ProjectionSemanticFragmentPhase5SubqueueDefinition,
    ProjectionSemanticFragmentPhase5TouchpointDefinition,
    iter_phase5_queues,
    iter_phase5_subqueues,
    iter_phase5_touchpoints,
)
from gabion.tooling.policy_substrate.site_identity import (
    canonical_site_identity,
    stable_hash,
)

_FORMAT_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES = frozenset(
    {
        "semantic_fragment.normalize_value",
        "semantic_fragment.stable_json_key",
        "projection_semantic_lowering.normalize_projection_op",
        "projection_semantic_lowering.lower_projection_op",
        "projection_semantic_lowering_compile.compile_semantic_projection_op",
        "projection_semantic_lowering_compile.semantic_rows_for_quotient_face",
        "projection_semantic_lowering_compile.semantic_rows_for_surface",
        "projection_exec.apply_execution_op",
        "projection_exec.sort_value",
        "projection_exec.canonical_group_reference",
    }
)


def _sorted[T](values: list[T], *, key=None) -> list[T]:
    return ordered_or_sorted(
        values,
        source="tooling.policy_substrate.invariant_graph",
        key=key,
    )


@dataclass(frozen=True)
class InvariantGraphNode:
    node_id: str
    node_kind: str
    title: str
    marker_name: str
    marker_kind: str
    marker_id: str
    site_identity: str
    structural_identity: str
    object_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]
    policy_ids: tuple[str, ...]
    invariant_ids: tuple[str, ...]
    reasoning_summary: str
    reasoning_control: str
    blocking_dependencies: tuple[str, ...]
    rel_path: str
    qualname: str
    line: int
    column: int
    ast_node_kind: str
    seam_class: str
    source_marker_node_id: str

    def matches_raw_id(self, raw_id: str) -> bool:
        values = {
            self.node_id,
            self.marker_id,
            self.site_identity,
            self.structural_identity,
            *self.object_ids,
            *self.doc_ids,
            *self.policy_ids,
            *self.invariant_ids,
        }
        return raw_id in values

    def as_payload(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "title": self.title,
            "marker_name": self.marker_name,
            "marker_kind": self.marker_kind,
            "marker_id": self.marker_id,
            "site_identity": self.site_identity,
            "structural_identity": self.structural_identity,
            "object_ids": list(self.object_ids),
            "doc_ids": list(self.doc_ids),
            "policy_ids": list(self.policy_ids),
            "invariant_ids": list(self.invariant_ids),
            "reasoning_summary": self.reasoning_summary,
            "reasoning_control": self.reasoning_control,
            "blocking_dependencies": list(self.blocking_dependencies),
            "rel_path": self.rel_path,
            "qualname": self.qualname,
            "line": self.line,
            "column": self.column,
            "ast_node_kind": self.ast_node_kind,
            "seam_class": self.seam_class,
            "source_marker_node_id": self.source_marker_node_id,
        }


@dataclass(frozen=True)
class InvariantGraphEdge:
    edge_id: str
    edge_kind: str
    source_id: str
    target_id: str

    def as_payload(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "edge_kind": self.edge_kind,
            "source_id": self.source_id,
            "target_id": self.target_id,
        }


@dataclass(frozen=True)
class InvariantGraphDiagnostic:
    diagnostic_id: str
    severity: str
    code: str
    node_id: str
    raw_dependency: str
    message: str

    def as_payload(self) -> dict[str, object]:
        return {
            "diagnostic_id": self.diagnostic_id,
            "severity": self.severity,
            "code": self.code,
            "node_id": self.node_id,
            "raw_dependency": self.raw_dependency,
            "message": self.message,
        }


@dataclass(frozen=True)
class InvariantGraph:
    root: str
    nodes: tuple[InvariantGraphNode, ...]
    edges: tuple[InvariantGraphEdge, ...]
    diagnostics: tuple[InvariantGraphDiagnostic, ...]

    def node_by_id(self) -> dict[str, InvariantGraphNode]:
        return {node.node_id: node for node in self.nodes}

    def edges_from(self) -> dict[str, tuple[InvariantGraphEdge, ...]]:
        grouped: defaultdict[str, list[InvariantGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            grouped[edge.source_id].append(edge)
        return {
            key: tuple(_sorted(values, key=lambda item: (item.edge_kind, item.target_id)))
            for key, values in grouped.items()
        }

    def edges_to(self) -> dict[str, tuple[InvariantGraphEdge, ...]]:
        grouped: defaultdict[str, list[InvariantGraphEdge]] = defaultdict(list)
        for edge in self.edges:
            grouped[edge.target_id].append(edge)
        return {
            key: tuple(_sorted(values, key=lambda item: (item.edge_kind, item.source_id)))
            for key, values in grouped.items()
        }

    def as_payload(self) -> dict[str, object]:
        node_kind_counts: dict[str, int] = defaultdict(int)
        edge_kind_counts: dict[str, int] = defaultdict(int)
        for node in self.nodes:
            node_kind_counts[node.node_kind] += 1
        for edge in self.edges:
            edge_kind_counts[edge.edge_kind] += 1
        return {
            "format_version": _FORMAT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": self.root,
            "counts": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "diagnostic_count": len(self.diagnostics),
                "node_kind_counts": dict(_sorted(list(node_kind_counts.items()))),
                "edge_kind_counts": dict(_sorted(list(edge_kind_counts.items()))),
            },
            "nodes": [node.as_payload() for node in self.nodes],
            "edges": [edge.as_payload() for edge in self.edges],
            "diagnostics": [item.as_payload() for item in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> InvariantGraph:
        nodes = tuple(
            InvariantGraphNode(
                node_id=str(item.get("node_id", "")),
                node_kind=str(item.get("node_kind", "")),
                title=str(item.get("title", "")),
                marker_name=str(item.get("marker_name", "")),
                marker_kind=str(item.get("marker_kind", "")),
                marker_id=str(item.get("marker_id", "")),
                site_identity=str(item.get("site_identity", "")),
                structural_identity=str(item.get("structural_identity", "")),
                object_ids=tuple(str(value) for value in item.get("object_ids", [])),
                doc_ids=tuple(str(value) for value in item.get("doc_ids", [])),
                policy_ids=tuple(str(value) for value in item.get("policy_ids", [])),
                invariant_ids=tuple(str(value) for value in item.get("invariant_ids", [])),
                reasoning_summary=str(item.get("reasoning_summary", "")),
                reasoning_control=str(item.get("reasoning_control", "")),
                blocking_dependencies=tuple(
                    str(value) for value in item.get("blocking_dependencies", [])
                ),
                rel_path=str(item.get("rel_path", "")),
                qualname=str(item.get("qualname", "")),
                line=int(item.get("line", 0)),
                column=int(item.get("column", 0)),
                ast_node_kind=str(item.get("ast_node_kind", "")),
                seam_class=str(item.get("seam_class", "")),
                source_marker_node_id=str(item.get("source_marker_node_id", "")),
            )
            for item in payload.get("nodes", [])
            if isinstance(item, Mapping)
        )
        edges = tuple(
            InvariantGraphEdge(
                edge_id=str(item.get("edge_id", "")),
                edge_kind=str(item.get("edge_kind", "")),
                source_id=str(item.get("source_id", "")),
                target_id=str(item.get("target_id", "")),
            )
            for item in payload.get("edges", [])
            if isinstance(item, Mapping)
        )
        diagnostics = tuple(
            InvariantGraphDiagnostic(
                diagnostic_id=str(item.get("diagnostic_id", "")),
                severity=str(item.get("severity", "")),
                code=str(item.get("code", "")),
                node_id=str(item.get("node_id", "")),
                raw_dependency=str(item.get("raw_dependency", "")),
                message=str(item.get("message", "")),
            )
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping)
        )
        return cls(
            root=str(payload.get("root", "")),
            nodes=nodes,
            edges=edges,
            diagnostics=diagnostics,
        )


def _synthetic_identity(
    *,
    ref_kind: str,
    value: str,
) -> tuple[str, str]:
    return (
        canonical_site_identity(
            rel_path="<synthetic>",
            qualname=f"{ref_kind}:{value}",
            line=0,
            column=0,
            node_kind="synthetic",
            surface="invariant_graph",
        ),
        canonical_structural_identity(
            rel_path="<synthetic>",
            qualname=f"{ref_kind}:{value}",
            structural_path=f"{ref_kind}:{value}",
            node_kind="synthetic",
            surface="invariant_graph",
        ),
    )


def _edge_id(edge_kind: str, source_id: str, target_id: str) -> str:
    return stable_hash("invariant_graph_edge", edge_kind, source_id, target_id)


def _marker_node_id(entry: InvariantMarkerScanNode) -> str:
    return f"{entry.scan_kind}:{entry.structural_identity}"


def _synthetic_node(
    *,
    node_id: str,
    title: str,
    ref_kind: str,
    value: str,
    object_ids: tuple[str, ...] = (),
    doc_ids: tuple[str, ...] = (),
    policy_ids: tuple[str, ...] = (),
    invariant_ids: tuple[str, ...] = (),
    reasoning_summary: str = "",
    reasoning_control: str = "",
    blocking_dependencies: tuple[str, ...] = (),
    rel_path: str = "",
    qualname: str = "",
    line: int = 0,
    column: int = 0,
    seam_class: str = "",
    marker_id: str = "",
    marker_name: str = "",
    marker_kind: str = "",
    source_marker_node_id: str = "",
    node_kind: str = "synthetic_work_item",
) -> InvariantGraphNode:
    site_identity, structural_identity = _synthetic_identity(
        ref_kind=ref_kind,
        value=value,
    )
    return InvariantGraphNode(
        node_id=node_id,
        node_kind=node_kind,
        title=title,
        marker_name=marker_name,
        marker_kind=marker_kind,
        marker_id=marker_id,
        site_identity=site_identity,
        structural_identity=structural_identity,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
        reasoning_summary=reasoning_summary,
        reasoning_control=reasoning_control,
        blocking_dependencies=blocking_dependencies,
        rel_path=rel_path,
        qualname=qualname,
        line=line,
        column=column,
        ast_node_kind="synthetic",
        seam_class=seam_class,
        source_marker_node_id=source_marker_node_id,
    )


def _node_from_scan_entry(entry: InvariantMarkerScanNode) -> InvariantGraphNode:
    return InvariantGraphNode(
        node_id=_marker_node_id(entry),
        node_kind=entry.scan_kind,
        title=entry.qualname or entry.marker_name,
        marker_name=entry.marker_name,
        marker_kind=entry.marker_kind,
        marker_id=entry.marker_id,
        site_identity=entry.site_identity,
        structural_identity=entry.structural_identity,
        object_ids=entry.object_ids,
        doc_ids=entry.doc_ids,
        policy_ids=entry.policy_ids,
        invariant_ids=entry.invariant_ids,
        reasoning_summary=entry.reasoning_summary,
        reasoning_control=entry.reasoning_control,
        blocking_dependencies=entry.blocking_dependencies,
        rel_path=entry.rel_path,
        qualname=entry.qualname,
        line=entry.line,
        column=entry.column,
        ast_node_kind=entry.ast_node_kind,
        seam_class="",
        source_marker_node_id="",
    )


def _synthetic_ref_node_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _primary_object_id(node: InvariantGraphNode) -> str:
    _prefix, _separator, primary_object_id = node.node_id.partition(":")
    if primary_object_id:
        return primary_object_id
    if node.object_ids:
        return node.object_ids[0]
    return node.title


def _semantic_ref_node(
    *,
    ref_kind: SemanticLinkKind,
    value: str,
) -> InvariantGraphNode:
    node_id = _synthetic_ref_node_id(ref_kind.value, value)
    object_ids = (value,) if ref_kind is SemanticLinkKind.OBJECT_ID else ()
    doc_ids = (value,) if ref_kind is SemanticLinkKind.DOC_ID else ()
    policy_ids = (value,) if ref_kind is SemanticLinkKind.POLICY_ID else ()
    invariant_ids = (value,) if ref_kind is SemanticLinkKind.INVARIANT_ID else ()
    return _synthetic_node(
        node_id=node_id,
        title=f"{ref_kind.value}:{value}",
        ref_kind=ref_kind.value,
        value=value,
        object_ids=object_ids,
        doc_ids=doc_ids,
        policy_ids=policy_ids,
        invariant_ids=invariant_ids,
    )


def _touchsite_seam_class(*, qualname: str, boundary_name: str) -> str:
    function_name = qualname.rsplit(".", 1)[-1]
    if (
        not function_name.startswith("_")
        or boundary_name in _PHASE5_SURVIVING_TOUCHSITE_BOUNDARY_NAMES
    ):
        return "surviving_carrier_seam"
    return "collapsible_helper_seam"


@dataclass(frozen=True)
class _Phase5Touchsite:
    touchsite_object_id: str
    touchpoint_object_id: str
    subqueue_object_id: str
    rel_path: str
    qualname: str
    boundary_name: str
    line: int
    column: int
    node_kind: str
    site_identity: str
    structural_identity: str
    seam_class: str


class _Phase5TouchsiteScanner(ast.NodeVisitor):
    def __init__(
        self,
        *,
        rel_path: str,
        touchpoint_object_id: str,
        subqueue_object_id: str,
    ) -> None:
        self.rel_path = rel_path
        self.touchpoint_object_id = touchpoint_object_id
        self.subqueue_object_id = subqueue_object_id
        self._scope: list[str] = []
        self.touchsites: list[_Phase5Touchsite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node=node, node_kind="function_def")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node=node, node_kind="async_function_def")

    def _visit_function(
        self,
        *,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        node_kind: str,
    ) -> None:
        self._scope.append(node.name)
        qualname = ".".join(self._scope)
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _dotted_name(decorator.func) != "grade_boundary":
                continue
            if _keyword_string_literal(decorator, "kind") != "semantic_carrier_adapter":
                continue
            boundary_name = _keyword_string_literal(decorator, "name") or qualname
            line = int(node.lineno)
            column = int(node.col_offset) + 1
            site_identity = canonical_site_identity(
                rel_path=self.rel_path,
                qualname=qualname,
                line=line,
                column=column,
                node_kind=node_kind,
                surface="semantic_carrier_adapter",
            )
            structural_identity = canonical_structural_identity(
                rel_path=self.rel_path,
                qualname=qualname,
                structural_path=f"{qualname}::grade_boundary[{boundary_name}]",
                node_kind=node_kind,
                surface="semantic_carrier_adapter",
            )
            seam_class = _touchsite_seam_class(
                qualname=qualname,
                boundary_name=boundary_name,
            )
            touchsite_object_id = f"PSF-007-TS:{structural_identity}"
            self.touchsites.append(
                _Phase5Touchsite(
                    touchsite_object_id=touchsite_object_id,
                    touchpoint_object_id=self.touchpoint_object_id,
                    subqueue_object_id=self.subqueue_object_id,
                    rel_path=self.rel_path,
                    qualname=qualname,
                    boundary_name=boundary_name,
                    line=line,
                    column=column,
                    node_kind=node_kind,
                    site_identity=site_identity,
                    structural_identity=structural_identity,
                    seam_class=seam_class,
                )
            )
        self.generic_visit(node)
        self._scope.pop()


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


def _keyword_string_literal(call: ast.Call, key: str) -> str:
    for keyword in call.keywords:
        if keyword.arg != key:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return str(keyword.value.value).strip()
    return ""


def _scan_phase5_touchsites(
    *,
    rel_path: str,
    touchpoint_object_id: str,
    subqueue_object_id: str,
) -> tuple[_Phase5Touchsite, ...]:
    source = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    scanner = _Phase5TouchsiteScanner(
        rel_path=rel_path,
        touchpoint_object_id=touchpoint_object_id,
        subqueue_object_id=subqueue_object_id,
    )
    scanner.visit(ast.parse(source, filename=rel_path))
    return tuple(
        _sorted(
            scanner.touchsites,
            key=lambda item: (
                item.rel_path,
                item.line,
                item.column,
                item.qualname,
                item.boundary_name,
            ),
        )
    )


def _phase5_primary_object_id(values: tuple[str, ...], fallback: str) -> str:
    for value in values:
        if value == fallback:
            return value
    return fallback


def build_invariant_graph(root: Path) -> InvariantGraph:
    root = root.resolve()
    nodes_by_id: dict[str, InvariantGraphNode] = {}
    diagnostics: list[InvariantGraphDiagnostic] = []
    edge_keys: set[tuple[str, str, str]] = set()
    edges: list[InvariantGraphEdge] = []
    object_node_ids: dict[str, str] = {}
    object_owner_node_ids: dict[str, str] = {}

    def add_node(node: InvariantGraphNode, *, replace: bool = False) -> None:
        existing = nodes_by_id.get(node.node_id)
        if existing is not None and not replace:
            raise ValueError(f"duplicate invariant graph node id: {node.node_id}")
        nodes_by_id[node.node_id] = node

    def add_edge(edge_kind: str, source_id: str, target_id: str) -> None:
        key = (edge_kind, source_id, target_id)
        if source_id == target_id or key in edge_keys:
            return
        edge_keys.add(key)
        edges.append(
            InvariantGraphEdge(
                edge_id=_edge_id(edge_kind, source_id, target_id),
                edge_kind=edge_kind,
                source_id=source_id,
                target_id=target_id,
            )
        )

    def ensure_ref_node(ref_kind: SemanticLinkKind, value: str) -> str:
        node_id = _synthetic_ref_node_id(ref_kind.value, value)
        if node_id not in nodes_by_id:
            add_node(_semantic_ref_node(ref_kind=ref_kind, value=value))
        if ref_kind is SemanticLinkKind.OBJECT_ID:
            object_node_ids.setdefault(value, node_id)
        return node_id

    def claim_object_ids(node: InvariantGraphNode) -> None:
        if node.node_kind not in {"synthetic_work_item", "synthetic_touchsite"}:
            return
        _prefix, _separator, primary_object_id = node.node_id.partition(":")
        if not primary_object_id:
            return
        claimed_by = object_owner_node_ids.get(primary_object_id)
        if claimed_by is not None and claimed_by != node.node_id:
            raise ValueError(
                "duplicate invariant graph object_id ownership: "
                f"{primary_object_id} claimed by {claimed_by} and {node.node_id}"
            )
        object_owner_node_ids[primary_object_id] = node.node_id

    marker_nodes = scan_invariant_markers(root)
    marker_node_id_by_marker_id: dict[str, str] = {}
    for marker_node in marker_nodes:
        graph_node = _node_from_scan_entry(marker_node)
        add_node(graph_node)
        marker_node_id_by_marker_id[graph_node.marker_id] = graph_node.node_id
        for link_value in graph_node.object_ids:
            add_edge(
                "links_to",
                graph_node.node_id,
                ensure_ref_node(SemanticLinkKind.OBJECT_ID, link_value),
            )
        for link_value in graph_node.doc_ids:
            add_edge(
                "links_to",
                graph_node.node_id,
                ensure_ref_node(SemanticLinkKind.DOC_ID, link_value),
            )
        for link_value in graph_node.policy_ids:
            add_edge(
                "links_to",
                graph_node.node_id,
                ensure_ref_node(SemanticLinkKind.POLICY_ID, link_value),
            )
        for link_value in graph_node.invariant_ids:
            add_edge(
                "links_to",
                graph_node.node_id,
                ensure_ref_node(SemanticLinkKind.INVARIANT_ID, link_value),
            )

    queue_definitions = tuple(iter_phase5_queues())
    subqueue_definitions = tuple(iter_phase5_subqueues())
    touchpoint_definitions = tuple(iter_phase5_touchpoints())
    subqueue_by_id = {item.subqueue_id: item for item in subqueue_definitions}
    touchpoint_by_id = {item.touchpoint_id: item for item in touchpoint_definitions}

    def _work_item_node(
        *,
        object_id: str,
        title: str,
        rel_path: str,
        qualname: str,
        line: int,
        marker_id: str,
        marker_name: str,
        marker_kind: str,
        site_identity: str,
        structural_identity: str,
        object_ids: tuple[str, ...],
        doc_ids: tuple[str, ...],
        policy_ids: tuple[str, ...],
        invariant_ids: tuple[str, ...],
        reasoning_summary: str,
        reasoning_control: str,
        blocking_dependencies: tuple[str, ...],
        source_marker_node_id: str,
    ) -> InvariantGraphNode:
        return InvariantGraphNode(
            node_id=_synthetic_ref_node_id("object_id", object_id),
            node_kind="synthetic_work_item",
            title=title,
            marker_name=marker_name,
            marker_kind=marker_kind,
            marker_id=marker_id,
            site_identity=site_identity,
            structural_identity=structural_identity,
            object_ids=object_ids,
            doc_ids=doc_ids,
            policy_ids=policy_ids,
            invariant_ids=invariant_ids,
            reasoning_summary=reasoning_summary,
            reasoning_control=reasoning_control,
            blocking_dependencies=blocking_dependencies,
            rel_path=rel_path,
            qualname=qualname,
            line=line,
            column=1,
            ast_node_kind="function_def",
            seam_class="",
            source_marker_node_id=source_marker_node_id,
        )

    for queue_definition in queue_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            queue_definition.queue_id,
        )
        node = _work_item_node(
            object_id=primary_object_id,
            title=queue_definition.title,
            rel_path=queue_definition.rel_path,
            qualname=queue_definition.qualname,
            line=queue_definition.line,
            marker_id=queue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=queue_definition.marker_payload.marker_kind.value,
            site_identity=queue_definition.site_identity,
            structural_identity=queue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in queue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=queue_definition.marker_payload.reasoning.summary,
            reasoning_control=queue_definition.marker_payload.reasoning.control,
            blocking_dependencies=queue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                queue_definition.marker_identity,
                "",
            ),
        )
        add_node(node, replace=True)
        claim_object_ids(node)
        object_node_ids[primary_object_id] = node.node_id
        for item in (*node.doc_ids, *node.policy_ids, *node.invariant_ids):
            kind = (
                SemanticLinkKind.DOC_ID
                if item in node.doc_ids
                else SemanticLinkKind.POLICY_ID
                if item in node.policy_ids
                else SemanticLinkKind.INVARIANT_ID
            )
            add_edge("links_to", node.node_id, ensure_ref_node(kind, item))

    for subqueue_definition in subqueue_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
                and link.value.startswith(subqueue_definition.subqueue_id)
            ),
            subqueue_definition.subqueue_id,
        )
        node = _work_item_node(
            object_id=primary_object_id,
            title=subqueue_definition.title,
            rel_path=subqueue_definition.rel_path,
            qualname=subqueue_definition.qualname,
            line=subqueue_definition.line,
            marker_id=subqueue_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=subqueue_definition.marker_payload.marker_kind.value,
            site_identity=subqueue_definition.site_identity,
            structural_identity=subqueue_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in subqueue_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=subqueue_definition.marker_payload.reasoning.summary,
            reasoning_control=subqueue_definition.marker_payload.reasoning.control,
            blocking_dependencies=subqueue_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                subqueue_definition.marker_identity,
                "",
            ),
        )
        add_node(node, replace=True)
        claim_object_ids(node)
        object_node_ids[primary_object_id] = node.node_id

    for touchpoint_definition in touchpoint_definitions:
        primary_object_id = _phase5_primary_object_id(
            tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
                and link.value.startswith(touchpoint_definition.touchpoint_id)
            ),
            touchpoint_definition.touchpoint_id,
        )
        node = _work_item_node(
            object_id=primary_object_id,
            title=touchpoint_definition.title,
            rel_path=touchpoint_definition.rel_path,
            qualname=touchpoint_definition.qualname,
            line=touchpoint_definition.line,
            marker_id=touchpoint_definition.marker_identity,
            marker_name="gabion.invariants.todo_decorator",
            marker_kind=touchpoint_definition.marker_payload.marker_kind.value,
            site_identity=touchpoint_definition.site_identity,
            structural_identity=touchpoint_definition.structural_identity,
            object_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.OBJECT_ID
            ),
            doc_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.DOC_ID
            ),
            policy_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.POLICY_ID
            ),
            invariant_ids=tuple(
                link.value
                for link in touchpoint_definition.marker_payload.links
                if link.kind is SemanticLinkKind.INVARIANT_ID
            ),
            reasoning_summary=touchpoint_definition.marker_payload.reasoning.summary,
            reasoning_control=touchpoint_definition.marker_payload.reasoning.control,
            blocking_dependencies=touchpoint_definition.marker_payload.reasoning.blocking_dependencies,
            source_marker_node_id=marker_node_id_by_marker_id.get(
                touchpoint_definition.marker_identity,
                "",
            ),
        )
        add_node(node, replace=True)
        claim_object_ids(node)
        object_node_ids[primary_object_id] = node.node_id

    for node in list(nodes_by_id.values()):
        for value in node.object_ids:
            add_edge(
                "links_to",
                node.node_id,
                ensure_ref_node(SemanticLinkKind.OBJECT_ID, value),
            )
        for value in node.doc_ids:
            add_edge(
                "links_to",
                node.node_id,
                ensure_ref_node(SemanticLinkKind.DOC_ID, value),
            )
        for value in node.policy_ids:
            add_edge(
                "links_to",
                node.node_id,
                ensure_ref_node(SemanticLinkKind.POLICY_ID, value),
            )
        for value in node.invariant_ids:
            add_edge(
                "links_to",
                node.node_id,
                ensure_ref_node(SemanticLinkKind.INVARIANT_ID, value),
            )

    for queue_definition in queue_definitions:
        queue_id = object_node_ids[queue_definition.queue_id]
        for subqueue_id in queue_definition.subqueue_ids:
            subqueue_node_id = object_node_ids[subqueue_id]
            add_edge("contains", queue_id, subqueue_node_id)
            add_edge("blocks", subqueue_node_id, queue_id)

    for subqueue_definition in subqueue_definitions:
        subqueue_node_id = object_node_ids[subqueue_definition.subqueue_id]
        for touchpoint_id in subqueue_definition.touchpoint_ids:
            touchpoint_node_id = object_node_ids[touchpoint_id]
            add_edge("contains", subqueue_node_id, touchpoint_node_id)
            add_edge("blocks", touchpoint_node_id, subqueue_node_id)

    for touchpoint_definition in touchpoint_definitions:
        touchpoint_node_id = object_node_ids[touchpoint_definition.touchpoint_id]
        touchsites = _scan_phase5_touchsites(
            rel_path=touchpoint_definition.rel_path,
            touchpoint_object_id=touchpoint_definition.touchpoint_id,
            subqueue_object_id=touchpoint_definition.subqueue_id,
        )
        for touchsite in touchsites:
            node_id = _synthetic_ref_node_id("object_id", touchsite.touchsite_object_id)
            add_node(
                InvariantGraphNode(
                    node_id=node_id,
                    node_kind="synthetic_touchsite",
                    title=touchsite.boundary_name,
                    marker_name="grade_boundary",
                    marker_kind="",
                    marker_id="",
                    site_identity=touchsite.site_identity,
                    structural_identity=touchsite.structural_identity,
                    object_ids=(touchsite.touchsite_object_id,),
                    doc_ids=(),
                    policy_ids=(),
                    invariant_ids=(),
                    reasoning_summary="PSF-007 touchsite remains active.",
                    reasoning_control=touchpoint_by_id[
                        touchpoint_definition.touchpoint_id
                    ].marker_payload.reasoning.control,
                    blocking_dependencies=(touchpoint_definition.touchpoint_id,),
                    rel_path=touchsite.rel_path,
                    qualname=touchsite.qualname,
                    line=touchsite.line,
                    column=touchsite.column,
                    ast_node_kind=touchsite.node_kind,
                    seam_class=touchsite.seam_class,
                    source_marker_node_id=object_node_ids[touchpoint_definition.touchpoint_id],
                ),
                replace=True,
            )
            claim_object_ids(nodes_by_id[node_id])
            object_node_ids[touchsite.touchsite_object_id] = node_id
            add_edge("contains", touchpoint_node_id, node_id)
            add_edge("blocks", node_id, touchpoint_node_id)

    for node in list(nodes_by_id.values()):
        for dependency in node.blocking_dependencies:
            target_id = object_node_ids.get(dependency)
            if target_id is None:
                diagnostics.append(
                    InvariantGraphDiagnostic(
                        diagnostic_id=stable_hash(
                            "invariant_graph_diagnostic",
                            node.node_id,
                            dependency,
                        ),
                        severity="warning",
                        code="unresolved_blocking_dependency",
                        node_id=node.node_id,
                        raw_dependency=dependency,
                        message=(
                            f"{node.node_id} declares blocking dependency {dependency} "
                            "that does not resolve to a graph object_id node."
                        ),
                    )
                )
                continue
            add_edge("depends_on", node.node_id, target_id)

    return InvariantGraph(
        root=str(root),
        nodes=tuple(
            _sorted(
                list(nodes_by_id.values()),
                key=lambda item: (item.node_kind, item.rel_path, item.line, item.node_id),
            )
        ),
        edges=tuple(
            _sorted(
                edges,
                key=lambda item: (item.edge_kind, item.source_id, item.target_id),
            )
        ),
        diagnostics=tuple(
            _sorted(
                diagnostics,
                key=lambda item: (item.severity, item.code, item.node_id, item.raw_dependency),
            )
        ),
    )


def build_psf_phase5_projection(
    graph: InvariantGraph,
    *,
    queue_object_id: str = "PSF-007",
) -> dict[str, object]:
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    queue_node = node_by_id[_synthetic_ref_node_id("object_id", queue_object_id)]
    subqueue_nodes = [
        node_by_id[edge.target_id]
        for edge in edges_from.get(queue_node.node_id, ())
        if edge.edge_kind == "contains"
    ]
    touchpoint_nodes: list[InvariantGraphNode] = []
    touchsite_nodes: list[InvariantGraphNode] = []
    subqueue_payloads: list[dict[str, object]] = []
    touchpoint_payloads: list[dict[str, object]] = []
    for subqueue_node in subqueue_nodes:
        touchpoints = [
            node_by_id[edge.target_id]
            for edge in edges_from.get(subqueue_node.node_id, ())
            if edge.edge_kind == "contains"
        ]
        touchpoint_ids = [_primary_object_id(node) for node in touchpoints]
        touchpoint_nodes.extend(touchpoints)
        touchsite_count = 0
        collapsible_count = 0
        surviving_count = 0
        for touchpoint_node in touchpoints:
            touchsites = [
                node_by_id[edge.target_id]
                for edge in edges_from.get(touchpoint_node.node_id, ())
                if edge.edge_kind == "contains"
            ]
            touchsite_nodes.extend(touchsites)
            touchsite_count += len(touchsites)
            collapsible_count += sum(
                1 for node in touchsites if node.seam_class == "collapsible_helper_seam"
            )
            surviving_count += sum(
                1 for node in touchsites if node.seam_class == "surviving_carrier_seam"
            )
        subqueue_payloads.append(
            {
                "subqueue_id": _primary_object_id(subqueue_node),
                "title": subqueue_node.title,
                "site_identity": subqueue_node.site_identity,
                "structural_identity": subqueue_node.structural_identity,
                "marker_identity": subqueue_node.marker_id,
                "marker_reason": subqueue_node.reasoning_summary or subqueue_node.title,
                "reasoning_summary": subqueue_node.reasoning_summary,
                "reasoning_control": subqueue_node.reasoning_control,
                "blocking_dependencies": list(subqueue_node.blocking_dependencies),
                "object_ids": list(subqueue_node.object_ids),
                "touchpoint_ids": touchpoint_ids,
                "touchsite_count": touchsite_count,
                "collapsible_touchsite_count": collapsible_count,
                "surviving_touchsite_count": surviving_count,
            }
        )
    for touchpoint_node in touchpoint_nodes:
        touchsites = [
            node_by_id[edge.target_id]
            for edge in edges_from.get(touchpoint_node.node_id, ())
            if edge.edge_kind == "contains"
        ]
        collapsible_count = sum(
            1 for node in touchsites if node.seam_class == "collapsible_helper_seam"
        )
        touchpoint_payloads.append(
            {
                "touchpoint_id": _primary_object_id(touchpoint_node),
                "subqueue_id": next(
                    (
                        _primary_object_id(subqueue_node)
                        for subqueue_node in subqueue_nodes
                        if any(
                            edge.target_id == touchpoint_node.node_id and edge.edge_kind == "contains"
                            for edge in edges_from.get(subqueue_node.node_id, ())
                        )
                    ),
                    "",
                ),
                "title": touchpoint_node.title,
                "rel_path": touchpoint_node.rel_path,
                "site_identity": touchpoint_node.site_identity,
                "structural_identity": touchpoint_node.structural_identity,
                "marker_identity": touchpoint_node.marker_id,
                "marker_reason": touchpoint_node.reasoning_summary or touchpoint_node.title,
                "reasoning_summary": touchpoint_node.reasoning_summary,
                "reasoning_control": touchpoint_node.reasoning_control,
                "blocking_dependencies": list(touchpoint_node.blocking_dependencies),
                "object_ids": list(touchpoint_node.object_ids),
                "touchsite_count": len(touchsites),
                "collapsible_touchsite_count": collapsible_count,
                "surviving_touchsite_count": len(touchsites) - collapsible_count,
                "touchsites": [
                    {
                        "touchsite_id": _primary_object_id(node),
                        "touchpoint_id": _primary_object_id(touchpoint_node),
                        "subqueue_id": next(
                            (
                                _primary_object_id(subqueue_node)
                                for subqueue_node in subqueue_nodes
                                if any(
                                    edge.target_id == touchpoint_node.node_id and edge.edge_kind == "contains"
                                    for edge in edges_from.get(subqueue_node.node_id, ())
                                )
                            ),
                            "",
                        ),
                        "rel_path": node.rel_path,
                        "qualname": node.qualname,
                        "boundary_name": node.title,
                        "line": node.line,
                        "column": node.column,
                        "node_kind": node.ast_node_kind,
                        "site_identity": node.site_identity,
                        "structural_identity": node.structural_identity,
                        "seam_class": node.seam_class,
                        "touchpoint_marker_identity": touchpoint_node.marker_id,
                        "touchpoint_structural_identity": touchpoint_node.structural_identity,
                        "subqueue_marker_identity": next(
                            (
                                subqueue_node.marker_id
                                for subqueue_node in subqueue_nodes
                                if any(
                                    edge.target_id == touchpoint_node.node_id and edge.edge_kind == "contains"
                                    for edge in edges_from.get(subqueue_node.node_id, ())
                                )
                            ),
                            "",
                        ),
                        "subqueue_structural_identity": next(
                            (
                                subqueue_node.structural_identity
                                for subqueue_node in subqueue_nodes
                                if any(
                                    edge.target_id == touchpoint_node.node_id and edge.edge_kind == "contains"
                                    for edge in edges_from.get(subqueue_node.node_id, ())
                                )
                            ),
                            "",
                        ),
                        "object_ids": list(node.object_ids),
                    }
                    for node in _sorted(
                        touchsites,
                        key=lambda item: (
                            item.rel_path,
                            item.line,
                            item.column,
                            item.qualname,
                        ),
                    )
                ],
            }
        )
    collapsible_touchsite_count = sum(
        1 for node in touchsite_nodes if node.seam_class == "collapsible_helper_seam"
    )
    return {
        "queue_id": queue_object_id,
        "title": queue_node.title,
        "remaining_touchsite_count": len(touchsite_nodes),
        "collapsible_touchsite_count": collapsible_touchsite_count,
        "surviving_touchsite_count": len(touchsite_nodes) - collapsible_touchsite_count,
        "subqueues": _sorted(subqueue_payloads, key=lambda item: str(item["subqueue_id"])),
        "touchpoints": _sorted(
            touchpoint_payloads,
            key=lambda item: str(item["touchpoint_id"]),
        ),
    }


def trace_nodes(graph: InvariantGraph, raw_id: str) -> tuple[InvariantGraphNode, ...]:
    nodes = [node for node in graph.nodes if node.matches_raw_id(raw_id)]
    return tuple(_sorted(nodes, key=lambda item: (item.node_kind, item.node_id)))


def blocker_chains(
    graph: InvariantGraph,
    *,
    object_id: str,
) -> dict[str, list[list[str]]]:
    queue_node_id = _synthetic_ref_node_id("object_id", object_id)
    node_by_id = graph.node_by_id()
    edges_from = graph.edges_from()
    if queue_node_id not in node_by_id:
        return {}
    grouped: defaultdict[str, list[list[str]]] = defaultdict(list)
    for subqueue_edge in edges_from.get(queue_node_id, ()):
        if subqueue_edge.edge_kind != "contains":
            continue
        subqueue_node = node_by_id[subqueue_edge.target_id]
        for touchpoint_edge in edges_from.get(subqueue_node.node_id, ()):
            if touchpoint_edge.edge_kind != "contains":
                continue
            touchpoint_node = node_by_id[touchpoint_edge.target_id]
            for touchsite_edge in edges_from.get(touchpoint_node.node_id, ()):
                if touchsite_edge.edge_kind != "contains":
                    continue
                touchsite_node = node_by_id[touchsite_edge.target_id]
                grouped[touchsite_node.seam_class or "unknown"].append(
                    [
                        queue_node_id,
                        subqueue_node.node_id,
                        touchpoint_node.node_id,
                        touchsite_node.node_id,
                    ]
                )
    return {
        key: _sorted(value, key=lambda item: tuple(item))
        for key, value in grouped.items()
    }


def load_invariant_graph(path: Path) -> InvariantGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return InvariantGraph.from_payload(payload)


def write_invariant_graph(path: Path, graph: InvariantGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.as_payload(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "InvariantGraph",
    "InvariantGraphDiagnostic",
    "InvariantGraphEdge",
    "InvariantGraphNode",
    "blocker_chains",
    "build_invariant_graph",
    "build_psf_phase5_projection",
    "load_invariant_graph",
    "trace_nodes",
    "write_invariant_graph",
]
