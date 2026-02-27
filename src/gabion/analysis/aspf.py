# gabion:boundary_normalization_module
from __future__ import annotations
# gabion:decision_protocol_module

from dataclasses import dataclass, field
import hashlib
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from gabion.analysis.aspf_evidence import normalize_alt_evidence_payload
from gabion.invariants import never
from gabion.order_contract import sort_once
from gabion.runtime import stable_encode


NodeKey = tuple[object, ...]
NodeFingerprint = tuple[str, tuple[str, ...]]
StructuralKeyAtom = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | tuple["StructuralKeyAtom", ...]
)


def _fingerprint_part(part: object) -> str:
    match part:
        case str() as text:
            return f"str:{text}"
        case bool() as flag:
            return f"bool:{flag}"
        case int() as number:
            return f"int:{number}"
        case float() as number:
            return f"float:{number!r}"
        case None:
            return "none:null"
        case _:
            return f"repr:{part!r}"


def fingerprint_identity(kind: str, key: NodeKey) -> NodeFingerprint:
    return (kind, tuple(_fingerprint_part(part) for part in key))


@dataclass(frozen=True)
class _InternIdentity:
    node_id: NodeId
    fingerprint: NodeFingerprint


def _canonicalize_intern_identity(kind: str, key: NodeKey) -> _InternIdentity:
    node_id = NodeId(kind=kind, key=key)
    return _InternIdentity(node_id=node_id, fingerprint=fingerprint_identity(kind, key))


@dataclass(frozen=True)
class NodeId:
    kind: str
    key: NodeKey

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "key": list(self.key)}

    def sort_key(self) -> tuple[str, tuple[str, ...]]:
        return (self.kind, tuple(str(part) for part in self.key))

    def fingerprint(self) -> NodeFingerprint:
        return fingerprint_identity(self.kind, self.key)


@dataclass(frozen=True)
class Node:
    node_id: NodeId
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.node_id.kind

    def as_dict(self) -> dict[str, object]:
        # Lazy import avoids module-cycle during timeout_context bootstrap.
        from gabion.analysis.timeout_context import check_deadline

        check_deadline()
        payload = self.node_id.as_dict()
        if self.meta:
            payload["meta"] = self.meta
        return payload


@dataclass(frozen=True)
class Alt:
    kind: str
    inputs: tuple[NodeId, ...]
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence",
            _canonicalize_evidence(self.evidence),
        )

    def as_dict(self) -> dict[str, object]:
        # Lazy import avoids module-cycle during timeout_context bootstrap.
        from gabion.analysis.timeout_context import check_deadline

        check_deadline()
        inputs: list[dict[str, object]] = []
        for node_id in self.inputs:
            check_deadline()
            inputs.append(node_id.as_dict())
        payload: dict[str, object] = {
            "kind": self.kind,
            "inputs": inputs,
        }
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload

    def sort_key(self) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
        # Lazy import avoids module-cycle during timeout_context bootstrap.
        from gabion.analysis.timeout_context import check_deadline

        check_deadline()
        inputs: list[tuple[str, tuple[str, ...]]] = []
        for node_id in self.inputs:
            check_deadline()
            inputs.append(node_id.sort_key())
        return (
            self.kind,
            tuple(inputs),
        )


def canon_param(name: str) -> str:
    return name.strip()


def canon_paramset(params: Iterable[str]) -> tuple[str, ...]:
    cleaned = {canon_param(p) for p in params if canon_param(p)}
    return tuple(sort_once(cleaned, source = 'src/gabion/analysis/aspf.py:123'))


def _canonicalize_evidence(evidence: object) -> dict[str, object]:
    normalized = normalize_alt_evidence_payload(evidence)
    return cast(dict[str, object], _canonicalize_evidence_value(normalized))


def _canonicalize_evidence_value(value: object) -> object:
    match value:
        case dict() as mapping:
            ordered_keys = sort_once(
                (str(key) for key in mapping),
                source="aspf._canonicalize_evidence_value.mapping",
            )
            return {
                key: _canonicalize_evidence_value(mapping[key])
                for key in ordered_keys
            }
        case list() | tuple() | set() as seq:
            canonical_items = [_canonicalize_evidence_value(item) for item in seq]
            by_stable_text: dict[str, object] = {}
            for item in canonical_items:
                stable_text = stable_encode.stable_compact_text(item)
                by_stable_text.setdefault(stable_text, item)
            return [
                by_stable_text[key]
                for key in sort_once(
                    by_stable_text,
                    source="aspf._canonicalize_evidence_value.sequence",
                )
            ]
        case _:
            return value


def _float_structural_atom(value: float) -> StructuralKeyAtom:
    if math.isnan(value):
        return ("float_nan",)
    if math.isinf(value):
        return ("float_inf", 1 if value > 0 else -1)
    numerator, denominator = value.as_integer_ratio()
    return ("float_ratio", numerator, denominator)


def structural_key_atom(
    value: object,
    *,
    source: str,
) -> StructuralKeyAtom:
    match value:
        case None:
            return ("none",)
        case bool() as bool_value:
            return ("bool", bool_value)
        case int() as int_value:
            return ("int", int_value)
        case float() as float_value:
            return _float_structural_atom(float_value)
        case str() as text_value:
            return ("str", text_value)
        case bytes() as bytes_value:
            return ("bytes", bytes_value)
        case Path() as path_value:
            return ("path", str(path_value))
        case NodeId() as node_id:
            return (
                "node_id",
                node_id.kind,
                tuple(
                    structural_key_atom(part, source=f"{source}.node_key")
                    for part in node_id.key
                ),
            )
        case Mapping() as mapping_value:
            items = [(str(key), mapping_value[key]) for key in mapping_value]
            ordered_items = sort_once(
                items,
                source=f"{source}.mapping_items",
                # Lexical mapping-key order defines canonical mapping identity.
                key=lambda item: item[0],
            )
            return (
                "map",
                tuple(
                    (
                        key,
                        structural_key_atom(raw_value, source=f"{source}.{key}"),
                    )
                    for key, raw_value in ordered_items
                ),
            )
        case list() as list_value:
            return (
                "list",
                tuple(
                    structural_key_atom(item, source=f"{source}.list_item")
                    for item in list_value
                ),
            )
        case tuple() as tuple_value:
            return (
                "tuple",
                tuple(
                    structural_key_atom(item, source=f"{source}.tuple_item")
                    for item in tuple_value
                ),
            )
        case set() as set_value:
            normalized_items = [
                structural_key_atom(item, source=f"{source}.set_item")
                for item in set_value
            ]
            ordered_items = sort_once(
                normalized_items,
                source=f"{source}.set_items",
                # Non-lexical comparator: typed structural atom tuple.
                key=lambda item: item,
            )
            return (
                "set",
                tuple(ordered_items),
            )
        case frozenset() as frozenset_value:
            normalized_items = [
                structural_key_atom(item, source=f"{source}.frozenset_item")
                for item in frozenset_value
            ]
            ordered_items = sort_once(
                normalized_items,
                source=f"{source}.frozenset_items",
                # Non-lexical comparator: typed structural atom tuple.
                key=lambda item: item,
            )
            return (
                "frozenset",
                tuple(ordered_items),
            )
        case _:
            never(
                "unsupported structural identity value",
                source=source,
                value_type=type(value).__name__,
            )


def structural_key_json(
    value: StructuralKeyAtom,
) -> object:
    match value:
        case tuple() as tuple_value:
            return [structural_key_json(entry) for entry in tuple_value]
        case bytes() as bytes_value:
            return {"_py": "bytes", "hex": bytes_value.hex()}
        case _:
            return value


@dataclass
class Forest:
    nodes: dict[NodeId, Node] = field(default_factory=dict)
    alts: list[Alt] = field(default_factory=list)
    _nodes_by_fingerprint: dict[NodeFingerprint, NodeId] = field(default_factory=dict)
    _alt_index: dict[
        tuple[str, tuple[NodeId, ...], StructuralKeyAtom],
        Alt,
    ] = field(default_factory=dict)

    def _intern_node(self, node_id: NodeId, meta: object) -> NodeId:
        identity = _canonicalize_intern_identity(node_id.kind, node_id.key)
        existing = self._nodes_by_fingerprint.get(identity.fingerprint)
        if existing is not None:
            return existing
        self.nodes[identity.node_id] = Node(node_id=identity.node_id, meta=meta or {})
        self._nodes_by_fingerprint[identity.fingerprint] = identity.node_id
        return identity.node_id

    def has_node(self, kind: str, key: NodeKey) -> bool:
        identity = _canonicalize_intern_identity(kind, key)
        return identity.fingerprint in self._nodes_by_fingerprint

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

    def add_site(self, path: str, qual: str, span: object = None) -> NodeId:
        file_id = self.add_file_site(path)
        key: NodeKey = (path, qual)
        if span is not None:
            normalized_span = cast(tuple[int, int, int, int], span)
            key = (*key, *normalized_span)
        node_id = NodeId(kind="FunctionSite", key=key)
        meta: dict[str, object] = {"path": path, "qual": qual}
        if span is not None:
            meta["span"] = list(cast(tuple[int, int, int, int], span))
        existed = self.has_node(node_id.kind, node_id.key)
        site_id = self._intern_node(node_id, meta)
        if not existed:
            self.add_alt("FunctionSiteInFile", (site_id, file_id), evidence={"path": path})
        return site_id

    def add_suite_site(
        self,
        path: str,
        qual: str,
        suite_kind: str,
        span: object = None,
        *,
        parent: object = None,
    ) -> NodeId:
        # dataflow-bundle: path, qual, suite_kind
        file_id = self.add_file_site(path)
        key: NodeKey = (path, qual, suite_kind)
        if span is not None:
            normalized_span = cast(tuple[int, int, int, int], span)
            key = (*key, *normalized_span)
        node_id = NodeId(kind="SuiteSite", key=key)
        meta: dict[str, object] = {
            "path": path,
            "qual": qual,
            "suite_kind": suite_kind,
            "suite_id": self._suite_site_id(
                path=path,
                qual=qual,
                suite_kind=suite_kind,
                span=span,
            ),
        }
        if span is not None:
            meta["span"] = list(cast(tuple[int, int, int, int], span))
        existed = self.has_node(node_id.kind, node_id.key)
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
                    cast(NodeId, parent),
                    suite_id,
                    evidence={"suite_kind": suite_kind},
                )
        return suite_id

    @staticmethod
    def _suite_site_id(
        *,
        path: str,
        qual: str,
        suite_kind: str,
        span: object,
    ) -> str:
        # dataflow-bundle: path, qual, suite_kind
        payload: dict[str, object] = {
            "domain": "python",
            "kind": str(suite_kind),
            "path": str(path),
            "qual": str(qual),
        }
        if span is not None:
            payload["span"] = [
                int(part) for part in cast(tuple[int, int, int, int], span)
            ]
        canonical = stable_encode.stable_compact_text(payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"suite:{digest}"

    def add_suite_contains(
        self,
        parent: NodeId,
        child: NodeId,
        evidence: object = None,
    ) -> Alt:
        # dataflow-bundle: child, parent
        return self.add_alt("SuiteContains", (parent, child), evidence=evidence)

    def add_spec_site(
        self,
        *,
        spec_hash: str,
        spec_name: str,
        spec_domain: object = None,
        spec_version: object = None,
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
            meta["spec_domain"] = str(spec_domain)
        if type(spec_version) is int:
            meta["spec_version"] = int(spec_version)
        return self._intern_node(node_id, meta)

    def add_alt(
        self,
        kind: str,
        inputs: Iterable[NodeId],
        evidence: object = None,
    ) -> Alt:
        # Forest mutation is semantic work and must run under an explicit
        # deadline clock scope.
        self._require_deadline_clock_scope("Forest.add_alt")
        # Lazy import avoids module-cycle during timeout_context bootstrap.
        from gabion.analysis.timeout_context import consume_deadline_ticks

        consume_deadline_ticks()
        normalized_kind = str(kind).strip()
        normalized_inputs = tuple(inputs)
        normalized_evidence = _canonicalize_evidence(evidence)
        evidence_identity = structural_key_atom(
            normalized_evidence,
            source="Forest.add_alt.evidence",
        )
        structural_key = (normalized_kind, normalized_inputs, evidence_identity)
        interned = self._alt_index.get(structural_key)
        if interned is not None:
            return interned
        alt = Alt(
            kind=normalized_kind,
            inputs=normalized_inputs,
            evidence=normalized_evidence,
        )
        self.alts.append(alt)
        self._alt_index[structural_key] = alt
        return alt

    @staticmethod
    def _require_deadline_clock_scope(operation: str) -> None:
        # Lazy import avoids module-cycle during timeout_context bootstrap.
        from gabion.analysis.timeout_context import get_deadline_clock
        from gabion.exceptions import NeverThrown
        from gabion.invariants import never

        try:
            get_deadline_clock()
        except NeverThrown:
            never(
                "forest mutation requires deadline_clock_scope",
                operation=operation,
            )

    def add_node(self, kind: str, key: NodeKey, meta: object = None) -> NodeId:
        node_id = NodeId(kind=kind, key=key)
        return self._intern_node(node_id, meta)

    def to_json(self) -> dict[str, object]:
        nodes = sort_once(self.nodes.values(), key=lambda node: node.node_id.sort_key(), source = 'src/gabion/analysis/aspf.py:346')
        alts = sort_once(self.alts, key=lambda alt: alt.sort_key(), source = 'src/gabion/analysis/aspf.py:347')
        return {
            "format_version": 1,
            "nodes": [node.as_dict() for node in nodes],
            "alts": [alt.as_dict() for alt in alts],
        }
