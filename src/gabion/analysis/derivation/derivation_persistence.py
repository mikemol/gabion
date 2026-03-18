# gabion:ambiguity_boundary_module
# gabion:boundary_normalization_module
# gabion:grade_boundary kind=semantic_carrier_adapter name=derivation_persistence
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import cast

from gabion.analysis.derivation.derivation_cache import DerivationCacheRuntime
from gabion.analysis.derivation.derivation_contract import DerivationOp
from gabion.analysis.derivation.derivation_graph import DerivationGraph
from gabion.analysis.aspf import aspf
from gabion.json_types import JSONValue
DERIVATION_CACHE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class MissingDerivationCheckpoint:
    kind: str = "missing"


@dataclass(frozen=True)
class InvalidDerivationCheckpoint:
    kind: str = "invalid"


@dataclass(frozen=True)
class LoadedDerivationCheckpoint:
    runtime_payload: dict[str, JSONValue]
    kind: str = "loaded"


DerivationCheckpointRead = (
    MissingDerivationCheckpoint | InvalidDerivationCheckpoint | LoadedDerivationCheckpoint
)


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
) -> DerivationCheckpointRead:
    if not path.exists():
        return MissingDerivationCheckpoint()
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return InvalidDerivationCheckpoint()
    match raw_payload:
        case dict() as checkpoint_payload:
            if (
                checkpoint_payload.get("format_version")
                != DERIVATION_CACHE_FORMAT_VERSION
            ):
                return InvalidDerivationCheckpoint()
            runtime_payload = checkpoint_payload.get("runtime")
            match runtime_payload:
                case dict() as runtime_map:
                    return LoadedDerivationCheckpoint(runtime_payload=runtime_map)
                case _:
                    return InvalidDerivationCheckpoint()
        case _:
            return InvalidDerivationCheckpoint()


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
            pass
        case _:
            return 0
    restored = 0
    for node_payload in filter(_is_hydratable_checkpoint_node_payload, node_payloads):
        op_payload = node_payload["op"]
        input_nodes_payload = node_payload["input_nodes"]
        graph.intern_derived(
            op=_derivation_op_from_payload(op_payload),
            input_nodes=tuple(
                _node_id_from_payload(raw_input) for raw_input in input_nodes_payload
            ),
            params=node_payload.get("params"),
            dependencies=node_payload.get("dependencies"),
            source="derivation_persistence.hydrate",
        )
        restored += 1
    return restored


def _is_node_id_payload(payload: object) -> bool:
    match payload:
        case Mapping() as payload_map:
            kind = payload_map.get("kind")
            return isinstance(kind, str) and bool(kind.strip())
        case _:
            return False


def _is_hydratable_checkpoint_node_payload(payload: object) -> bool:
    match payload:
        case Mapping() as node_payload:
            op_payload = node_payload.get("op")
            input_nodes_payload = node_payload.get("input_nodes")
            match (op_payload, input_nodes_payload):
                case (Mapping() as op_payload_map, list() as input_nodes_list):
                    name = op_payload_map.get("name")
                    return isinstance(name, str) and bool(name.strip()) and all(
                        _is_node_id_payload(raw_input) for raw_input in input_nodes_list
                    )
                case _:
                    return False
        case _:
            return False


def _derivation_op_from_payload(payload: Mapping[str, object]) -> DerivationOp:
    return DerivationOp(
        name=str(payload.get("name", "") or ""),
        version=int(payload.get("version", 1) or 1),
        scope=str(payload.get("scope", "analysis") or "analysis"),
    )


def _node_id_from_payload(payload: Mapping[str, JSONValue]) -> aspf.NodeId:
    key_atom = _structural_json_to_atom(payload.get("key"))
    normalized_key = key_atom if isinstance(key_atom, tuple) else (key_atom,)
    kind = cast(str, payload["kind"])
    return aspf.NodeId(kind=kind, key=normalized_key)


def _structural_json_to_atom(value: object) -> object:
    match value:
        case list() as sequence_value:
            return tuple(_structural_json_to_atom(entry) for entry in sequence_value)
        case Mapping() as mapping_value:
            raw_hex = mapping_value.get("hex")
            if mapping_value.get("_py") == "bytes" and isinstance(raw_hex, str):
                try:
                    return bytes.fromhex(raw_hex)
                except ValueError:
                    return b""
            return tuple(
                (
                    str(key),
                    _structural_json_to_atom(raw_value),
                )
                for key, raw_value in mapping_value.items()
            )
        case _:
            return value
