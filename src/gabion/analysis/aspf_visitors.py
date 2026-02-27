from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Mapping, Protocol, cast

from gabion.analysis.aspf import Alt, Forest, Node
from gabion.json_types import JSONObject, JSONValue
from gabion.order_contract import sort_once


class AspfTraversalVisitor(Protocol):
    """Semantic-core visitor contract for ASPF traversal dispatch.

    Boundary normalization is expected to happen at ingress parsers (for example,
    ``load_trace_payload``). Traversal helpers below assume normalized payloads and
    perform deterministic visitor dispatch for semantic-core consumers.
    """

    def on_forest_node(self, *, node: Node) -> None: ...

    def on_forest_alt(self, *, alt: Alt) -> None: ...

    def on_trace_one_cell(
        self,
        *,
        index: int,
        one_cell: Mapping[str, object],
    ) -> None: ...

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None: ...

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None: ...

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None: ...

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None: ...


@dataclass
class NullAspfTraversalVisitor:
    def on_forest_node(self, *, node: Node) -> None:
        pass

    def on_forest_alt(self, *, alt: Alt) -> None:
        pass

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        pass

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        pass

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        pass

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        pass

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None:
        pass


def traverse_forest_to_visitor(*, forest: Forest, visitor: AspfTraversalVisitor) -> None:
    nodes = sort_once(
        forest.nodes.values(),
        key=lambda node: node.node_id.sort_key(),
        source="aspf_visitors.traverse_forest_to_visitor.nodes",
    )
    for node in nodes:
        visitor.on_forest_node(node=node)
    alts = sort_once(
        forest.alts,
        key=lambda alt: alt.sort_key(),
        source="aspf_visitors.traverse_forest_to_visitor.alts",
    )
    for alt in alts:
        visitor.on_forest_alt(alt=alt)


def replay_trace_payload_to_visitor(
    *,
    trace_payload: Mapping[str, object],
    visitor: AspfTraversalVisitor,
) -> None:
    one_cells = cast(list[Mapping[str, object]], trace_payload.get("one_cells", []))
    for index, one_cell in enumerate(one_cells):
        visitor.on_trace_one_cell(index=index, one_cell=one_cell)

    two_cell_witnesses = cast(
        list[Mapping[str, object]],
        trace_payload.get("two_cell_witnesses", []),
    )
    for index, witness in enumerate(two_cell_witnesses):
        visitor.on_trace_two_cell_witness(index=index, witness=witness)

    cofibrations = cast(
        list[Mapping[str, object]],
        trace_payload.get("cofibration_witnesses", []),
    )
    for index, cofibration in enumerate(cofibrations):
        visitor.on_trace_cofibration(index=index, cofibration=cofibration)

    surface_payload = cast(
        Mapping[str, object],
        trace_payload.get("surface_representatives", {}),
    )
    ordered_surfaces = sort_once(
        [str(surface) for surface in surface_payload],
        source="aspf_visitors.replay_trace_payload_to_visitor.surface_representatives",
    )
    for surface in ordered_surfaces:
        visitor.on_trace_surface_representative(
            surface=surface,
            representative=str(surface_payload.get(surface, "")),
        )


def replay_equivalence_payload_to_visitor(
    *,
    equivalence_payload: Mapping[str, object],
    visitor: AspfTraversalVisitor,
) -> None:
    rows = cast(list[Mapping[str, object]], equivalence_payload.get("surface_table", []))
    for index, row in enumerate(rows):
        visitor.on_equivalence_surface_row(index=index, row=row)


@dataclass
class TracePayloadEmitter(NullAspfTraversalVisitor):
    one_cells: list[JSONValue] = field(default_factory=list)
    two_cell_witnesses: list[JSONValue] = field(default_factory=list)
    cofibration_witnesses: list[JSONValue] = field(default_factory=list)
    surface_representatives: dict[str, str] = field(default_factory=dict)

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        self.one_cells.append({str(key): cast(JSONValue, one_cell[key]) for key in one_cell})

    def on_trace_two_cell_witness(
        self,
        *,
        index: int,
        witness: Mapping[str, object],
    ) -> None:
        self.two_cell_witnesses.append(
            {str(key): cast(JSONValue, witness[key]) for key in witness}
        )

    def on_trace_cofibration(
        self,
        *,
        index: int,
        cofibration: Mapping[str, object],
    ) -> None:
        self.cofibration_witnesses.append(
            {str(key): cast(JSONValue, cofibration[key]) for key in cofibration}
        )

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        self.surface_representatives[str(surface)] = str(representative)


@dataclass
class OpportunityPayloadEmitter(NullAspfTraversalVisitor):
    _materialize_kinds_by_resume_ref: dict[str, set[str]] = field(default_factory=dict)
    _representative_to_surfaces: dict[str, list[str]] = field(default_factory=dict)
    _fungible_rows: list[JSONObject] = field(default_factory=list)

    def on_trace_one_cell(self, *, index: int, one_cell: Mapping[str, object]) -> None:
        kind = str(one_cell.get("kind", ""))
        metadata = cast(Mapping[str, object], one_cell.get("metadata", {}))
        resume_ref = ""
        for candidate_key in ("state_path", "import_state_path"):
            candidate = str(metadata.get(candidate_key, "")).strip()
            if candidate:
                resume_ref = candidate
                break
        if resume_ref:
            self._materialize_kinds_by_resume_ref.setdefault(resume_ref, set()).add(kind)

    def on_trace_surface_representative(
        self,
        *,
        surface: str,
        representative: str,
    ) -> None:
        self._representative_to_surfaces.setdefault(representative, []).append(surface)

    def on_equivalence_surface_row(
        self,
        *,
        index: int,
        row: Mapping[str, object],
    ) -> None:
        surface = str(row.get("surface", "")).strip()
        witness_id = row.get("witness_id")
        if (
            str(row.get("classification")) == "non_drift"
            and bool(surface)
            and witness_id not in (None, "")
        ):
            self._fungible_rows.append(
                {
                    "opportunity_id": f"opp:fungible-substitution:{surface}",
                    "kind": "fungible_execution_path_substitution",
                    "confidence": 0.82,
                    "affected_surfaces": [surface],
                    "witness_ids": [str(witness_id)],
                    "reason": "2-cell witness links baseline/current representatives",
                }
            )

    def build_rows(self) -> list[JSONObject]:
        rows: list[JSONObject] = []
        for resume_ref in sort_once(
            self._materialize_kinds_by_resume_ref,
            source="aspf_visitors.OpportunityPayloadEmitter.materialize.resume_ref",
        ):
            kinds = self._materialize_kinds_by_resume_ref[resume_ref]
            fused = int("resume_load" in kinds and "resume_write" in kinds)
            rows.append(
                {
                    "opportunity_id": f"opp:materialize-load-fusion:{resume_ref}",
                    "kind": ("materialize_load_observed", "materialize_load_fusion")[fused],
                    "confidence": (0.51, 0.74)[fused],
                    "affected_surfaces": [],
                    "witness_ids": [],
                    "reason": (
                        "resume boundary observed state reference",
                        "resume load and write boundaries share state reference",
                    )[fused],
                }
            )

        for representative in sort_once(
            self._representative_to_surfaces,
            source="aspf_visitors.OpportunityPayloadEmitter.representative",
        ):
            surfaces = tuple(
                sort_once(
                    self._representative_to_surfaces[representative],
                    source="aspf_visitors.OpportunityPayloadEmitter.representative_surfaces",
                )
            )
            digest = hashlib.sha256(representative.encode("utf-8")).hexdigest()[:12]
            rows.append(
                {
                    "opportunity_id": f"opp:reusable-boundary:{digest}",
                    "kind": "reusable_boundary_artifact",
                    "confidence": 0.67,
                    "affected_surfaces": list(surfaces),
                    "witness_ids": [],
                    "reason": "multiple semantic surfaces share deterministic representative",
                }
            )

        rows.extend(self._fungible_rows)
        return sorted(rows, key=lambda item: str(item.get("opportunity_id", "")))


@dataclass
class StatePayloadEmitter:
    trace: JSONObject = field(default_factory=dict)
    equivalence: JSONObject = field(default_factory=dict)
    opportunities: JSONObject = field(default_factory=dict)

    def set_trace_payload(self, payload: Mapping[str, object]) -> None:
        self.trace = {str(key): cast(JSONValue, payload[key]) for key in payload}

    def set_equivalence_payload(self, payload: Mapping[str, object]) -> None:
        self.equivalence = {str(key): cast(JSONValue, payload[key]) for key in payload}

    def set_opportunities_payload(self, payload: Mapping[str, object]) -> None:
        self.opportunities = {str(key): cast(JSONValue, payload[key]) for key in payload}
