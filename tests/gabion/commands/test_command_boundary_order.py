from __future__ import annotations

from pathlib import Path

import pytest

from gabion import server
from gabion.commands import boundary_order, command_ids, payload_codec
from gabion.exceptions import NeverThrown
from gabion.lsp_client import CommandRequest, run_command_direct


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_boundary_order_canonicalizes_nested_mapping_and_sequences::boundary_order.py::gabion.commands.boundary_order.canonicalize_boundary_mapping
def test_boundary_order_canonicalizes_nested_mapping_and_sequences() -> None:
    payload = {
        "z": [3, 1, 2],
        "a": {"beta": 2, "alpha": 1},
        "set_values": {"b", "a"},
        "tuple_values": ("z", "x", "y"),
        "dict_list": [{"k": "b", "v": 2}, {"v": 1, "k": "a"}],
    }
    ordered = boundary_order.canonicalize_boundary_mapping(
        payload,
        source="tests.test_command_boundary_order.payload",
    )

    assert list(ordered.keys()) == ["a", "dict_list", "set_values", "tuple_values", "z"]
    assert ordered["z"] == [1, 2, 3]
    assert ordered["set_values"] == ["a", "b"]
    assert ordered["tuple_values"] == ("x", "y", "z")
    assert ordered["dict_list"] == [{"k": "a", "v": 1}, {"k": "b", "v": 2}]


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_boundary_order_rejects_second_active_sort::boundary_order.py::gabion.commands.boundary_order.canonicalize_boundary_mapping
def test_boundary_order_rejects_second_active_sort() -> None:
    payload = {"b": 2, "a": 1}
    ordered = boundary_order.canonicalize_boundary_mapping(
        payload,
        source="tests.test_command_boundary_order.second_sort.first",
    )
    with pytest.raises(NeverThrown):
        boundary_order.canonicalize_boundary_mapping(
            ordered,
            source="tests.test_command_boundary_order.second_sort.second",
        )


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_apply_boundary_updates_once_uses_shared_mapping_comparator::boundary_order.py::gabion.commands.boundary_order.apply_boundary_updates_once
def test_apply_boundary_updates_once_uses_shared_mapping_comparator() -> None:
    base = boundary_order.canonicalize_boundary_mapping(
        {"z": 3, "m": 2},
        source="tests.test_command_boundary_order.apply_updates.base",
    )
    merged = boundary_order.apply_boundary_updates_once(
        base,
        {
            "a": {"beta": 2, "alpha": 1},
            "y": [3, 1, 2],
            "m": 5,
        },
        source="tests.test_command_boundary_order.apply_updates.merged",
    )

    assert list(merged.keys()) == ["a", "m", "y", "z"]
    assert merged["m"] == 5
    assert merged["y"] == [1, 2, 3]
    assert merged["a"] == {"alpha": 1, "beta": 2}


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_payload_codec_normalizes_ingress_payload_order::payload_codec.py::gabion.commands.payload_codec.normalized_command_payload
def test_payload_codec_normalizes_ingress_payload_order() -> None:
    raw_payload = {
        "paths": ["b.py", "a.py"],
        "exclude": ["z", "a"],
        "config": {"b": 2, "a": 1},
    }
    command_args, payload = payload_codec.normalized_command_payload(
        command=command_ids.DATAFLOW_COMMAND,
        arguments=[raw_payload],
    )

    assert isinstance(command_args[0], dict)
    assert payload == command_args[0]
    assert payload["paths"] == ["a.py", "b.py"]
    assert payload["exclude"] == ["a", "z"]
    assert payload["config"] == {"a": 1, "b": 2}


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_payload_codec_normalized_command_id_list_covers_list_and_default_paths::payload_codec.py::gabion.commands.payload_codec.normalized_command_id_list
def test_payload_codec_normalized_command_id_list_covers_list_and_default_paths() -> None:
    assert payload_codec.normalized_command_id_list(
        {"commands": ["gabion.z", "gabion.a"]},
        key="commands",
    ) == ("gabion.a", "gabion.z")
    assert payload_codec.normalized_command_id_list({}, key="commands") == ()


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_payload_codec_normalized_command_id_list_rejects_non_list::payload_codec.py::gabion.commands.payload_codec.normalized_command_id_list
def test_payload_codec_normalized_command_id_list_rejects_non_list() -> None:
    with pytest.raises(NeverThrown):
        payload_codec.normalized_command_id_list(
            {"commands": "gabion.check"},
            key="commands",
        )


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_server_require_payload_normalizes_order::server.py::gabion.server._require_payload
def test_server_require_payload_normalizes_order() -> None:
    payload = {
        "paths": ["c.py", "a.py", "b.py"],
        "section": {"z": 3, "a": 1},
    }
    ordered = server._require_payload(
        payload,
        command=command_ids.DATAFLOW_COMMAND,
    )

    assert list(ordered.keys()) == ["paths", "section"]
    assert ordered["paths"] == ["a.py", "b.py", "c.py"]
    assert ordered["section"] == {"a": 1, "z": 3}


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_lsp_direct_enforces_ingress_and_egress_order::lsp_client.py::gabion.lsp_client.run_command_direct
def test_lsp_direct_enforces_ingress_and_egress_order(tmp_path: Path) -> None:
    observed_notifications: list[dict[str, object]] = []
    captured_payload: dict[str, object] = {}

    def _fake_execute_command(ls, payload=None):
        assert isinstance(payload, dict)
        captured_payload.update(payload)
        ls.send_notification(
            "$/progress",
            {
                "token": "gabion.dataflowAudit/progress-v1",
                "value": {"items": [1, 2, 3], "metric": {"a": 1, "z": 2}},
            },
        )
        return {"a": {"a": 1, "z": 2}, "z": [1, 2, 3]}

    request = CommandRequest(
        command_ids.DATAFLOW_COMMAND,
        [
            {
                "analysis_timeout_ticks": 100,
                "analysis_timeout_tick_ns": 1_000_000,
                "paths": ["b.py", "a.py"],
            }
        ],
    )
    result = run_command_direct(
        request,
        root=tmp_path,
        notification_callback=observed_notifications.append,
        execute_dataflow_fn=_fake_execute_command,
    )

    assert captured_payload["paths"] == ["a.py", "b.py"]
    assert list(result.keys()) == ["a", "z"]
    assert result["z"] == [1, 2, 3]
    assert observed_notifications
    params = observed_notifications[0]["params"]
    assert isinstance(params, dict)
    value = params["value"]
    assert isinstance(value, dict)
    assert value["items"] == [1, 2, 3]
    metric = value["metric"]
    assert isinstance(metric, dict)
    assert list(metric.keys()) == ["a", "z"]


class _SortedCarrierList(list[object]):
    pass


class _TruthyEmptyUpdates(dict[str, object]):
    def __init__(self) -> None:
        super().__init__()
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        return super().__iter__()

    def __bool__(self) -> bool:
        return True


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_boundary_order_value_wrappers_and_update_shortcuts::boundary_order.py::gabion.commands.boundary_order.canonicalize_boundary_value::boundary_order.py::gabion.commands.boundary_order.normalize_boundary_value_once::boundary_order.py::gabion.commands.boundary_order.apply_boundary_updates_once::boundary_order.py::gabion.commands.boundary_order.enforce_boundary_value_ordered
def test_boundary_order_value_wrappers_and_update_shortcuts() -> None:
    assert boundary_order.canonicalize_boundary_value(
        {"b": 2, "a": 1},
        source="tests.test_command_boundary_order.value_wrapper",
    ) == {"a": 1, "b": 2}
    assert boundary_order.normalize_boundary_value_once(
        {"b": 2, "a": 1},
        source="tests.test_command_boundary_order.normalize_unsorted_value",
    ) == {"a": 1, "b": 2}

    sorted_list = _SortedCarrierList([1, 2, 3])
    sorted_list._gabion_sorted_once = True  # type: ignore[attr-defined]
    sorted_list._gabion_sort_source = "tests.test_command_boundary_order.sorted_list"  # type: ignore[attr-defined]
    assert boundary_order.normalize_boundary_value_once(
        sorted_list,
        source="tests.test_command_boundary_order.normalize_sorted_list",
    ) == [1, 2, 3]
    assert boundary_order.canonicalize_boundary_value(
        sorted_list,
        source="tests.test_command_boundary_order.canonicalize_sorted_list",
    ) == [1, 2, 3]
    assert boundary_order.enforce_boundary_value_ordered(
        {"a": 1, "b": 2},
        source="tests.test_command_boundary_order.enforce_value",
    ) == {"a": 1, "b": 2}

    base = boundary_order.canonicalize_boundary_mapping(
        {"a": 1},
        source="tests.test_command_boundary_order.base_mapping",
    )
    assert (
        boundary_order.apply_boundary_updates_once(
            base,
            {},
            source="tests.test_command_boundary_order.empty_updates",
        )
        == base
    )
    truthy_empty_updates = _TruthyEmptyUpdates()
    assert (
        boundary_order.apply_boundary_updates_once(
            base,
            truthy_empty_updates,
            source="tests.test_command_boundary_order.truthy_empty_updates",
        )
        == base
    )
    assert truthy_empty_updates.iteration_count == 1


# gabion:evidence E:call_footprint::tests/test_command_boundary_order.py::test_boundary_order_rejects_set_on_enforced_egress::boundary_order.py::gabion.commands.boundary_order.enforce_boundary_value_ordered
def test_boundary_order_rejects_set_on_enforced_egress() -> None:
    with pytest.raises(NeverThrown):
        boundary_order.enforce_boundary_value_ordered(
            {"b", "a"},
            source="tests.test_command_boundary_order.set_egress",
        )
