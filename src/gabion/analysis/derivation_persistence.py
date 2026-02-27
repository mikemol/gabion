# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

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
) -> object:
    if not path.exists():
        return None
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    match raw_payload:
        case dict() as checkpoint_payload:
            if (
                checkpoint_payload.get("format_version")
                != DERIVATION_CACHE_FORMAT_VERSION
            ):
                return None
            runtime_payload = checkpoint_payload.get("runtime")
            match runtime_payload:
                case dict() as runtime_map:
                    return runtime_map
                case _:
                    return None
        case _:
            return None


def hydrate_graph_from_checkpoint(
    *,
    graph: DerivationGraph,
    runtime_payload: Mapping[str, JSONValue],
) -> int:
    graph_payload = runtime_payload.get("graph")
    match graph_payload:
        case Mapping() as graph_payload_map:
            nodes = graph_payload_map.get("nodes")
        case _:
            return 0
    match nodes:
        case list() as node_payloads:
            nodes = node_payloads
        case _:
            return 0
    restored = 0
    for raw_node in nodes:
        match raw_node:
            case Mapping() as node_payload:
                op_payload = node_payload.get("op")
                input_nodes_payload = node_payload.get("input_nodes")
                params_payload = node_payload.get("params")
                dependencies_payload = node_payload.get("dependencies")
                match (op_payload, input_nodes_payload):
                    case (Mapping() as op_payload_map, list() as input_nodes_list):
                        pass
                    case _:
                        continue
            case _:
                continue
        op_name = str(op_payload_map.get("name", "") or "")
        if not op_name:
            continue
        op = DerivationOp(
            name=op_name,
            version=int(op_payload_map.get("version", 1) or 1),
            scope=str(op_payload_map.get("scope", "analysis") or "analysis"),
        )
        input_nodes = []
        invalid_input = False
        for raw_input in input_nodes_list:
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
) -> object:
    match payload:
        case Mapping() as payload_map:
            kind = str(payload_map.get("kind", "") or "")
            key_payload = payload_map.get("key")
            if not kind:
                return None
            key_atom = _structural_json_to_atom(key_payload)
            match key_atom:
                case tuple() as key_tuple:
                    normalized_key = key_tuple
                case _:
                    normalized_key = (key_atom,)
            return aspf.NodeId(kind=kind, key=normalized_key)
        case _:
            return None


def _structural_json_to_atom(value: object) -> object:
    match value:
        case list() as sequence_value:
            return tuple(_structural_json_to_atom(entry) for entry in sequence_value)
        case Mapping() as mapping_value:
            kind = mapping_value.get("_py")
            if kind == "bytes":
                raw_hex = mapping_value.get("hex")
                match raw_hex:
                    case str() as hex_text:
                        try:
                            return bytes.fromhex(hex_text)
                        except ValueError:
                            return b""
                    case _:
                        pass
            return tuple(
                (
                    str(key),
                    _structural_json_to_atom(raw_value),
                )
                for key, raw_value in mapping_value.items()
            )
        case _:
            return value
