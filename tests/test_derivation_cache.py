from __future__ import annotations

import math
from pathlib import Path

import pytest

from gabion.analysis import aspf, derivation_persistence
from gabion.analysis.derivation_cache import (
    DerivationCacheRuntime,
    get_global_derivation_cache,
    reset_global_derivation_cache,
)
from gabion.analysis.derivation_contract import DerivationOp
from gabion.analysis.derivation_persistence import (
    DERIVATION_CACHE_FORMAT_VERSION,
    hydrate_graph_from_checkpoint,
    read_derivation_checkpoint,
    write_derivation_checkpoint,
)
from gabion.exceptions import NeverThrown
from tests.env_helpers import env_scope


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_structural_key_atom_canonicalizes_mapping_order_and_preserves_list_order::aspf.py::gabion.analysis.aspf.structural_key_atom
def test_structural_key_atom_canonicalizes_mapping_order_and_preserves_list_order() -> None:
    first = aspf.structural_key_atom(
        {"b": 2, "a": [1, 2]},
        source="tests.structural.first",
    )
    second = aspf.structural_key_atom(
        {"a": [1, 2], "b": 2},
        source="tests.structural.second",
    )
    third = aspf.structural_key_atom(
        {"a": [2, 1], "b": 2},
        source="tests.structural.third",
    )

    assert first == second
    assert first != third


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_derivation_cache_hits_and_invalidation_regeneration::derivation_cache.py::gabion.analysis.derivation_cache.DerivationCacheRuntime.derive::derivation_cache.py::gabion.analysis.derivation_cache.DerivationCacheRuntime.invalidate
def test_derivation_cache_hits_and_invalidation_regeneration() -> None:
    runtime = DerivationCacheRuntime(max_entries=4)
    op = DerivationOp(name="demo.cache", version=1, scope="tests")
    calls = {"count": 0}

    def _compute() -> dict[str, int]:
        calls["count"] += 1
        return {"value": calls["count"]}

    first = runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=_compute,
        source="tests.test_derivation_cache.first",
    )
    second = runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=_compute,
        source="tests.test_derivation_cache.second",
    )

    assert first == {"value": 1}
    assert second == {"value": 1}
    assert calls["count"] == 1
    assert runtime.stats().hits == 1

    node_id = next(iter(runtime._values))
    runtime.invalidate(node_id)

    third = runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=_compute,
        source="tests.test_derivation_cache.third",
    )

    assert third == {"value": 2}
    assert calls["count"] == 2
    assert runtime.stats().regenerations >= 1


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_derivation_graph_interns_equivalent_inputs::derivation_graph.py::gabion.analysis.derivation_graph.DerivationGraph.intern_input
def test_derivation_graph_interns_equivalent_inputs() -> None:
    runtime = DerivationCacheRuntime(max_entries=4)
    first = runtime.graph.intern_input(
        input_label="payload",
        value={"b": 2, "a": 1},
        source="tests.graph.first",
    )
    second = runtime.graph.intern_input(
        input_label="payload",
        value={"a": 1, "b": 2},
        source="tests.graph.second",
    )

    assert first == second


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_structural_key_atom_covers_float_set_frozenset_and_nodeid::aspf.py::gabion.analysis.aspf.structural_key_atom::aspf.py::gabion.analysis.aspf.structural_key_json
def test_structural_key_atom_covers_float_set_frozenset_and_nodeid() -> None:
    nan_atom = aspf.structural_key_atom(math.nan, source="tests.structural.nan")
    inf_atom = aspf.structural_key_atom(math.inf, source="tests.structural.inf")
    finite_atom = aspf.structural_key_atom(1.5, source="tests.structural.finite")
    set_atom = aspf.structural_key_atom({"z", "a"}, source="tests.structural.set")
    frozenset_atom = aspf.structural_key_atom(
        frozenset({"z", "a"}),
        source="tests.structural.frozenset",
    )
    node_atom = aspf.structural_key_atom(
        aspf.NodeId(kind="Node", key=("x",)),
        source="tests.structural.node",
    )
    json_payload = aspf.structural_key_json(("bytes", b"ab"))

    assert nan_atom == ("float_nan",)
    assert inf_atom == ("float_inf", 1)
    assert finite_atom == ("float_ratio", 3, 2)
    assert set_atom == ("set", (("str", "a"), ("str", "z")))
    assert frozenset_atom == ("frozenset", (("str", "a"), ("str", "z")))
    assert node_atom[0] == "node_id"
    assert json_payload == ["bytes", {"_py": "bytes", "hex": "6162"}]


def test_structural_key_atom_covers_bytes_and_path(tmp_path: Path) -> None:
    path_value = tmp_path / "sample.py"
    bytes_atom = aspf.structural_key_atom(b"ab", source="tests.structural.bytes")
    path_atom = aspf.structural_key_atom(path_value, source="tests.structural.path")
    assert bytes_atom == ("bytes", b"ab")
    assert path_atom == ("path", str(path_value))


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_structural_json_to_atom_bytes_mapping_non_string_hex_falls_through_to_mapping_tuple::derivation_persistence.py::gabion.analysis.derivation_persistence._structural_json_to_atom
def test_structural_json_to_atom_bytes_mapping_non_string_hex_falls_through_to_mapping_tuple() -> None:
    value = {"_py": "bytes", "hex": 123}
    assert derivation_persistence._structural_json_to_atom(value) == (
        ("_py", "bytes"),
        ("hex", 123),
    )


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_structural_key_atom_rejects_unsupported_values::aspf.py::gabion.analysis.aspf.structural_key_atom
def test_structural_key_atom_rejects_unsupported_values() -> None:
    class Unsupported:
        pass

    with pytest.raises(NeverThrown):
        aspf.structural_key_atom(Unsupported(), source="tests.structural.unsupported")


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_derivation_cache_eviction_and_payload_and_callable_invalidation::derivation_cache.py::gabion.analysis.derivation_cache.DerivationCacheRuntime.to_payload::derivation_graph.py::gabion.analysis.derivation_graph.DerivationGraph.to_payload
def test_derivation_cache_eviction_and_payload_and_callable_invalidation() -> None:
    runtime = DerivationCacheRuntime(max_entries=1)
    op = DerivationOp(name="demo.payload", scope="tests", version=1)
    events: list[str] = []

    runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 1},
        source="tests.payload.first",
        on_cache_event=lambda event, _node_id: events.append(event),
    )
    runtime.derive(
        op=op,
        structural_inputs={"path": "b.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 2},
        source="tests.payload.second",
        on_cache_event=lambda event, _node_id: events.append(event),
    )
    assert runtime.stats().evictions == 1
    assert "cache:miss" in events

    invalidated = runtime.invalidate(lambda _node_id: True)
    assert invalidated
    assert runtime.stats().invalidations >= 1
    assert runtime.materialize(invalidated[0]) is None

    payload = runtime.to_payload()
    assert payload["format_version"] == 1
    assert isinstance(payload["cached_nodes"], list)
    assert isinstance(payload["graph"], dict)


def test_derivation_cache_events_materialize_and_iterable_inputs() -> None:
    runtime = DerivationCacheRuntime(max_entries=4)
    op = DerivationOp(name="demo.events", scope="tests", version=1)
    events: list[str] = []

    first = runtime.derive(
        op=op,
        structural_inputs=["a.py"],
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 1},
        source="tests.events.first",
        on_cache_event=lambda event, _node_id: events.append(event),
    )
    second = runtime.derive(
        op=op,
        structural_inputs=["a.py"],
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 2},
        source="tests.events.second",
        on_cache_event=lambda event, _node_id: events.append(event),
    )
    node_id = next(iter(runtime._values))
    assert first == {"value": 1}
    assert second == {"value": 1}
    assert runtime.materialize(node_id) == {"value": 1}
    runtime.invalidate(node_id)
    third = runtime.derive(
        op=op,
        structural_inputs=["a.py"],
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 3},
        source="tests.events.third",
        on_cache_event=lambda event, _node_id: events.append(event),
    )
    assert third == {"value": 3}
    assert "cache:hit" in events
    assert "cache:regen:start" in events
    assert "cache:regen:done" in events


def test_derivation_cache_invalidate_skips_non_cached_nodes() -> None:
    runtime = DerivationCacheRuntime(max_entries=4)
    op = DerivationOp(name="demo.invalidate", scope="tests", version=1)
    runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 1},
        source="tests.invalidate.first",
    )
    derived_node = next(iter(runtime._values))
    input_node = runtime.graph.dependencies_for(derived_node)[0]
    invalidated = runtime.invalidate(input_node)
    assert input_node in invalidated
    assert runtime.materialize(derived_node) is None


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_global_derivation_cache_helpers_reset_and_env_parse::derivation_cache.py::gabion.analysis.derivation_cache.reset_global_derivation_cache::derivation_cache.py::gabion.analysis.derivation_cache.get_global_derivation_cache
def test_global_derivation_cache_helpers_reset_and_env_parse() -> None:
    runtime = reset_global_derivation_cache(max_entries=3)
    assert runtime.max_entries == 3
    assert get_global_derivation_cache() is runtime


def test_global_derivation_cache_helpers_invalid_env_uses_default() -> None:
    with env_scope({"GABION_DERIVATION_CACHE_MAX_ENTRIES": "not-an-int"}):
        runtime = reset_global_derivation_cache()
    assert runtime.max_entries == 4096
    with env_scope({"GABION_DERIVATION_CACHE_MAX_ENTRIES": "2"}):
        runtime = reset_global_derivation_cache()
    assert runtime.max_entries == 2


# gabion:evidence E:call_footprint::tests/test_derivation_cache.py::test_derivation_persistence_roundtrip_and_hydrate::derivation_persistence.py::gabion.analysis.derivation_persistence.write_derivation_checkpoint::derivation_persistence.py::gabion.analysis.derivation_persistence.read_derivation_checkpoint::derivation_persistence.py::gabion.analysis.derivation_persistence.hydrate_graph_from_checkpoint
def test_derivation_persistence_roundtrip_and_hydrate(tmp_path: Path) -> None:
    runtime = DerivationCacheRuntime(max_entries=4)
    op = DerivationOp(name="demo.persist", version=1, scope="tests")
    runtime.derive(
        op=op,
        structural_inputs={"path": "a.py"},
        dependencies={"mtime_ns": 1},
        params={"kind": "demo"},
        compute_fn=lambda: {"value": 1},
        source="tests.persist.first",
    )
    checkpoint = tmp_path / "derivation.json"
    write_derivation_checkpoint(path=checkpoint, runtime=runtime)

    loaded = read_derivation_checkpoint(path=checkpoint)
    assert isinstance(loaded, dict)
    assert loaded.get("format_version") == 1

    restored = hydrate_graph_from_checkpoint(
        graph=DerivationCacheRuntime(max_entries=4).graph,
        runtime_payload=loaded,
    )
    assert restored >= 1

    checkpoint.write_text("{\"format_version\": 1}")
    assert read_derivation_checkpoint(path=checkpoint) is None

    checkpoint.write_text(
        f"{{\"format_version\": {DERIVATION_CACHE_FORMAT_VERSION}, \"runtime\": []}}"
    )
    assert read_derivation_checkpoint(path=checkpoint) is None


def test_derivation_persistence_read_checkpoint_invalid_inputs(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    assert read_derivation_checkpoint(path=missing_path) is None

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{", encoding="utf-8")
    assert read_derivation_checkpoint(path=bad_path) is None

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    assert read_derivation_checkpoint(path=list_path) is None


def test_derivation_persistence_hydrate_graph_guards() -> None:
    graph = DerivationCacheRuntime(max_entries=4).graph
    assert hydrate_graph_from_checkpoint(graph=graph, runtime_payload={"graph": []}) == 0
    assert hydrate_graph_from_checkpoint(graph=graph, runtime_payload={"graph": {"nodes": {}}}) == 0
    restored = hydrate_graph_from_checkpoint(
        graph=graph,
        runtime_payload={
            "graph": {
                "nodes": [
                    "not-a-mapping",
                    {"op": "bad", "input_nodes": []},
                    {"op": {"name": ""}, "input_nodes": []},
                    {"op": {"name": "x"}, "input_nodes": [{}]},
                ]
            }
        },
    )
    assert restored == 0


def test_derivation_persistence_node_payload_conversion_guards() -> None:
    assert derivation_persistence._node_id_from_payload("bad") is None
    assert derivation_persistence._node_id_from_payload({"key": "value"}) is None

    node_id = derivation_persistence._node_id_from_payload(
        {"kind": "Derivation", "key": "raw"}
    )
    assert node_id is not None
    assert node_id.key == ("raw",)
    assert derivation_persistence._structural_json_to_atom({"_py": "bytes", "hex": "61"}) == b"a"
    assert derivation_persistence._structural_json_to_atom({"_py": "bytes", "hex": "zz"}) == b""
    assert derivation_persistence._structural_json_to_atom({"left": ["x"]}) == (
        ("left", ("x",)),
    )


def test_derivation_graph_duplicate_edges_and_cycle_invalidation() -> None:
    graph = DerivationCacheRuntime(max_entries=4).graph
    input_a = graph.intern_input(input_label="a", value=1, source="tests.graph.a")
    op_left = DerivationOp(name="left", version=1, scope="tests")
    op_right = DerivationOp(name="right", version=1, scope="tests")
    op_top = DerivationOp(name="top", version=1, scope="tests")
    left = graph.intern_derived(
        op=op_left,
        input_nodes=(input_a,),
        params={"kind": "left"},
        dependencies={"dep": 1},
        source="tests.graph.left",
    )
    right = graph.intern_derived(
        op=op_right,
        input_nodes=(input_a,),
        params={"kind": "right"},
        dependencies={"dep": 1},
        source="tests.graph.right",
    )
    top = graph.intern_derived(
        op=op_top,
        input_nodes=(left, right),
        params={"kind": "top"},
        dependencies={"dep": 1},
        source="tests.graph.top",
    )

    graph.record_edge(input_node_id=input_a, output_node_id=left, op_label="left")
    unknown = aspf.NodeId(kind="Unknown", key=("x",))
    assert graph.dependencies_for(unknown) == ()
    invalidated = graph.invalidate(input_a)
    assert input_a in invalidated
    assert top in invalidated
