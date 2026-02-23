# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from gabion.analysis.derivation_cache import DerivationCacheRuntime
from gabion.analysis.derivation_contract import DerivationOp
from gabion.analysis.derivation_graph import DerivationGraph
from gabion.analysis import aspf
from gabion.json_types import JSONValue


DERIVATION_CACHE_FORMAT_VERSION = 2


def write_derivation_checkpoint(
    *,
    path: Path,
    runtime: DerivationCacheRuntime,
) -> None:
    payload = {
        "format_version": DERIVATION_CACHE_FORMAT_VERSION,
        "runtime": runtime.to_payload(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=False, separators=(",", ":"), ensure_ascii=True),
        encoding="utf-8",
    )


def read_derivation_checkpoint(
    *,
    path: Path,
) -> Mapping[str, JSONValue] | None:
    if not path.exists():
        return None
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_payload, dict):
        return None
    if raw_payload.get("format_version") != DERIVATION_CACHE_FORMAT_VERSION:
        return None
    runtime_payload = raw_payload.get("runtime")
    if not isinstance(runtime_payload, dict):
        return None
    return runtime_payload


def hydrate_graph_from_checkpoint(
    *,
    graph: DerivationGraph,
    runtime_payload: Mapping[str, JSONValue],
) -> int:
    graph_payload = runtime_payload.get("graph")
    if not isinstance(graph_payload, Mapping):
        return 0
    nodes = graph_payload.get("nodes")
    if not isinstance(nodes, list):
        return 0
    restored = 0
    for raw_node in nodes:
        if not isinstance(raw_node, Mapping):
            continue
        op_payload = raw_node.get("op")
        input_nodes_payload = raw_node.get("input_nodes")
        params_payload = raw_node.get("params")
        dependencies_payload = raw_node.get("dependencies")
        if (
            not isinstance(op_payload, Mapping)
            or not isinstance(input_nodes_payload, list)
        ):
            continue
        op_name = str(op_payload.get("name", "") or "")
        if not op_name:
            continue
        op = DerivationOp(
            name=op_name,
            version=int(op_payload.get("version", 1) or 1),
            scope=str(op_payload.get("scope", "analysis") or "analysis"),
        )
        input_nodes = []
        invalid_input = False
        for raw_input in input_nodes_payload:
            parsed_input = _node_id_from_payload(raw_input)
            if parsed_input is None:
                invalid_input = True
                break
            input_nodes.append(parsed_input)
        if invalid_input:
            continue
        graph.intern_derived(
            op=op,
            input_nodes=tuple(input_nodes),
            params=params_payload,
            dependencies=dependencies_payload,
            source="derivation_persistence.hydrate",
        )
        restored += 1
    return restored


def _node_id_from_payload(
    payload: object,
) -> aspf.NodeId | None:
    if not isinstance(payload, Mapping):
        return None
    kind = str(payload.get("kind", "") or "")
    key_payload = payload.get("key")
    if not kind:
        return None
    key_atom = _structural_json_to_atom(key_payload)
    if not isinstance(key_atom, tuple):
        key_atom = (key_atom,)
    return aspf.NodeId(kind=kind, key=key_atom)


def _structural_json_to_atom(value: object) -> object:
    if isinstance(value, list):
        return tuple(_structural_json_to_atom(entry) for entry in value)
    if isinstance(value, Mapping):
        kind = value.get("_py")
        if kind == "bytes":
            raw_hex = value.get("hex")
            if isinstance(raw_hex, str):
                try:
                    return bytes.fromhex(raw_hex)
                except ValueError:
                    return b""
        return tuple(
            (
                str(key),
                _structural_json_to_atom(raw_value),
            )
            for key, raw_value in value.items()
        )
    return value
