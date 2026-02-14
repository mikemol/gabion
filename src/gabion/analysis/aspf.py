from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


NodeKey = tuple[object, ...]


@dataclass(frozen=True)
class NodeId:
    kind: str
    key: NodeKey

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "key": list(self.key)}

    def sort_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.kind, tuple(str(part) for part in self.key))


@dataclass(frozen=True)
class Node:
    node_id: NodeId
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.node_id.kind

    def as_dict(self) -> dict[str, object]:
        payload = self.node_id.as_dict()
        if self.meta:
            payload["meta"] = self.meta
        return payload


@dataclass(frozen=True)
class Alt:
    kind: str
    inputs: tuple[NodeId, ...]
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "inputs": [node_id.as_dict() for node_id in self.inputs],
        }
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload

    def sort_key(self) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
        return (
            self.kind,
            tuple(node_id.sort_key() for node_id in self.inputs),
        )


def canon_param(name: str) -> str:
    return name.strip()


def canon_paramset(params: Iterable[str]) -> tuple[str, ...]:
    cleaned = {canon_param(p) for p in params if canon_param(p)}
    return tuple(sorted(cleaned))


@dataclass
class Forest:
    nodes: dict[NodeId, Node] = field(default_factory=dict)
    alts: list[Alt] = field(default_factory=list)

    def _intern_node(self, node_id: NodeId, meta: dict[str, object] | None) -> NodeId:
        if node_id in self.nodes:
            return node_id
        self.nodes[node_id] = Node(node_id=node_id, meta=meta or {})
        return node_id

    def add_file_site(self, path: str) -> NodeId:
        key = (path,)
        node_id = NodeId(kind="FileSite", key=key)
        return self._intern_node(node_id, {"path": path})

    def add_param(self, name: str) -> NodeId:
        key = (canon_param(name),)
        node_id = NodeId(kind="Param", key=key)
        return self._intern_node(node_id, {"name": key[0]})

    def add_paramset(self, params: Iterable[str]) -> NodeId:
        key = canon_paramset(params)
        node_id = NodeId(kind="ParamSet", key=key)
        self._intern_node(node_id, {"params": list(key)})
        if key:
            param_nodes = tuple(self.add_param(p) for p in key)
            self.add_alt("ParamSetMembers", (node_id, *param_nodes))
        return node_id

    def add_site(self, path: str, qual: str, span: tuple[int, int, int, int] | None = None) -> NodeId:
        file_id = self.add_file_site(path)
        key: NodeKey = (path, qual)
        if span is not None:
            key = (*key, *span)
        node_id = NodeId(kind="FunctionSite", key=key)
        meta: dict[str, object] = {"path": path, "qual": qual}
        if span is not None:
            meta["span"] = list(span)
        existed = node_id in self.nodes
        site_id = self._intern_node(node_id, meta)
        if not existed:
            self.add_alt("FunctionSiteInFile", (site_id, file_id), evidence={"path": path})
        return site_id

    def add_suite_site(
        self,
        path: str,
        qual: str,
        suite_kind: str,
        span: tuple[int, int, int, int] | None = None,
        *,
        parent: NodeId | None = None,
    ) -> NodeId:
        file_id = self.add_file_site(path)
        key: NodeKey = (path, qual, suite_kind)
        if span is not None:
            key = (*key, *span)
        node_id = NodeId(kind="SuiteSite", key=key)
        meta: dict[str, object] = {
            "path": path,
            "qual": qual,
            "suite_kind": suite_kind,
        }
        if span is not None:
            meta["span"] = list(span)
        existed = node_id in self.nodes
        suite_id = self._intern_node(node_id, meta)
        if not existed:
            func_id = self.add_site(path, qual)
            self.add_alt(
                "SuiteSiteInFunction",
                (suite_id, func_id),
                evidence={"suite_kind": suite_kind},
            )
            self.add_alt(
                "SuiteSiteInFile",
                (suite_id, file_id),
                evidence={"suite_kind": suite_kind},
            )
            if parent is not None:
                self.add_suite_contains(
                    parent,
                    suite_id,
                    evidence={"suite_kind": suite_kind},
                )
        return suite_id

    def add_suite_contains(
        self,
        parent: NodeId,
        child: NodeId,
        evidence: dict[str, object] | None = None,
    ) -> Alt:
        return self.add_alt("SuiteContains", (parent, child), evidence=evidence)

    def add_spec_site(
        self,
        *,
        spec_hash: str,
        spec_name: str,
        spec_domain: str | None = None,
        spec_version: int | None = None,
    ) -> NodeId:
        key: NodeKey = ("projection_spec", spec_hash, "spec")
        node_id = NodeId(kind="SuiteSite", key=key)
        meta: dict[str, object] = {
            "path": "projection_spec",
            "qual": spec_hash,
            "suite_kind": "spec",
            "spec_name": spec_name,
            "spec_hash": spec_hash,
        }
        if spec_domain:
            meta["spec_domain"] = spec_domain
        if spec_version is not None:
            meta["spec_version"] = spec_version
        return self._intern_node(node_id, meta)

    def add_alt(
        self,
        kind: str,
        inputs: Iterable[NodeId],
        evidence: dict[str, object] | None = None,
    ) -> Alt:
        alt = Alt(kind=kind, inputs=tuple(inputs), evidence=evidence or {})
        self.alts.append(alt)
        return alt

    def add_node(self, kind: str, key: NodeKey, meta: dict[str, object] | None = None) -> NodeId:
        node_id = NodeId(kind=kind, key=key)
        return self._intern_node(node_id, meta)

    def to_json(self) -> dict[str, object]:
        nodes = sorted(self.nodes.values(), key=lambda node: node.node_id.sort_key())
        alts = sorted(self.alts, key=lambda alt: alt.sort_key())
        return {
            "format_version": 1,
            "nodes": [node.as_dict() for node in nodes],
            "alts": [alt.as_dict() for alt in alts],
        }
