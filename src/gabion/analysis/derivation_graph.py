# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from gabion.analysis import aspf
from gabion.analysis.derivation_contract import (
    DerivationEdge,
    DerivationKey,
    DerivationNode,
    DerivationNodeId,
    DerivationOp,
    StructuralAtom,
)
from gabion.order_contract import sort_once


@dataclass
class DerivationGraph:
    forest: aspf.Forest = field(default_factory=aspf.Forest)
    nodes_by_id: dict[DerivationNodeId, DerivationNode] = field(default_factory=dict)
    edges_by_output: dict[DerivationNodeId, dict[DerivationNodeId, DerivationEdge]] = field(
        default_factory=dict
    )
    dependents_by_input: dict[DerivationNodeId, dict[DerivationNodeId, None]] = field(
        default_factory=dict
    )

    def intern_input(
        self,
        *,
        input_label: str,
        value: object,
        source: str,
    ) -> DerivationNodeId:
        structural_value = aspf.structural_key_atom(
            value,
            source=f"{source}.value",
        )
        node_id = self.forest.add_node(
            "DerivationInput",
            (
                input_label,
                structural_value,
            ),
            meta={
                "input_label": input_label,
            },
        )
        if node_id in self.nodes_by_id:
            return node_id
        key = DerivationKey(
            op=DerivationOp(name="input", scope=input_label, version=1),
            input_nodes=(),
            params=structural_value,
            dependencies=("none",),
        )
        self.nodes_by_id[node_id] = DerivationNode(
            node_id=node_id,
            key=key,
            source=source,
        )
        return node_id

    def intern_derived(
        self,
        *,
        op: DerivationOp,
        input_nodes: tuple[DerivationNodeId, ...],
        params: object,
        dependencies: object,
        source: str,
    ) -> DerivationNodeId:
        structural_params = aspf.structural_key_atom(
            params,
            source=f"{source}.params",
        )
        structural_dependencies = aspf.structural_key_atom(
            dependencies,
            source=f"{source}.dependencies",
        )
        key = DerivationKey(
            op=op,
            input_nodes=input_nodes,
            params=structural_params,
            dependencies=structural_dependencies,
        )
        node_id = self.forest.add_node(
            "Derivation",
            (
                op.as_key(),
                input_nodes,
                structural_params,
                structural_dependencies,
            ),
            meta={
                "op": op.name,
                "scope": op.scope,
                "version": op.version,
            },
        )
        if node_id in self.nodes_by_id:
            return node_id
        self.nodes_by_id[node_id] = DerivationNode(
            node_id=node_id,
            key=key,
            source=source,
        )
        for input_node in input_nodes:
            self.record_edge(
                input_node_id=input_node,
                output_node_id=node_id,
                op_label=op.name,
            )
        return node_id

    def record_edge(
        self,
        *,
        input_node_id: DerivationNodeId,
        output_node_id: DerivationNodeId,
        op_label: str,
    ) -> None:
        output_edges = self.edges_by_output.setdefault(output_node_id, {})
        if input_node_id in output_edges:
            return
        output_edges[input_node_id] = DerivationEdge(
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            op_label=op_label,
        )
        dependents = self.dependents_by_input.setdefault(input_node_id, {})
        dependents[output_node_id] = None

    def dependencies_for(
        self,
        node_id: DerivationNodeId,
    ) -> tuple[DerivationNodeId, ...]:
        edges = self.edges_by_output.get(node_id, {})
        return tuple(
            sort_once(
                edges,
                source="src/gabion/analysis/derivation_graph.py:dependencies_for",
                key=lambda value: value.sort_key(),
            )
        )

    def dependents_for(
        self,
        node_id: DerivationNodeId,
    ) -> tuple[DerivationNodeId, ...]:
        dependents = self.dependents_by_input.get(node_id, {})
        return tuple(
            sort_once(
                dependents,
                source="src/gabion/analysis/derivation_graph.py:dependents_for",
                key=lambda value: value.sort_key(),
            )
        )

    def invalidate(
        self,
        node_id: DerivationNodeId,
    ) -> tuple[DerivationNodeId, ...]:
        queue = [node_id]
        seen: dict[DerivationNodeId, None] = {}
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen[current] = None
            for dependent in self.dependents_for(current):
                queue.append(dependent)
        return tuple(
            sort_once(
                seen,
                source="src/gabion/analysis/derivation_graph.py:invalidate",
                key=lambda value: value.sort_key(),
            )
        )

    def to_payload(self) -> dict[str, object]:
        node_ids = sort_once(
            self.nodes_by_id,
            source="src/gabion/analysis/derivation_graph.py:to_payload.nodes",
            key=lambda value: value.sort_key(),
        )
        nodes_payload = []
        for node_id in node_ids:
            node = self.nodes_by_id[node_id]
            nodes_payload.append(
                {
                    "node_id": _node_id_payload(
                        node_id,
                        source="derivation_graph.to_payload.node_id",
                    ),
                    "op": {
                        "name": node.key.op.name,
                        "version": node.key.op.version,
                        "scope": node.key.op.scope,
                    },
                    "input_nodes": [
                        _node_id_payload(
                            entry,
                            source="derivation_graph.to_payload.input_nodes",
                        )
                        for entry in node.key.input_nodes
                    ],
                    "params": aspf.structural_key_json(node.key.params),
                    "dependencies": aspf.structural_key_json(node.key.dependencies),
                    "source": node.source,
                }
            )
        edges_payload = []
        for node_id in node_ids:
            edges = self.edges_by_output.get(node_id, {})
            edge_nodes = sort_once(
                edges,
                source="src/gabion/analysis/derivation_graph.py:to_payload.edges",
                key=lambda value: value.sort_key(),
            )
            for input_node in edge_nodes:
                edge = edges[input_node]
                edges_payload.append(
                    {
                        "input": _node_id_payload(
                            edge.input_node_id,
                            source="derivation_graph.to_payload.edge_input",
                        ),
                        "output": _node_id_payload(
                            edge.output_node_id,
                            source="derivation_graph.to_payload.edge_output",
                        ),
                        "op": edge.op_label,
                    }
                )
        return {
            "format_version": 1,
            "nodes": nodes_payload,
            "edges": edges_payload,
        }


def dependency_token(
    dependencies: Mapping[str, object],
    *,
    source: str,
) -> StructuralAtom:
    return aspf.structural_key_atom(
        dependencies,
        source=source,
    )


def _node_id_payload(
    node_id: DerivationNodeId,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "kind": node_id.kind,
        "key": aspf.structural_key_json(
            aspf.structural_key_atom(
                node_id.key,
                source=source,
            )
        ),
    }
