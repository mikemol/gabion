from __future__ import annotations

import pytest

from gabion.commands import boundary_order
from gabion.exceptions import NeverThrown
from gabion.order_contract import enforce_ordered, sort_once
from gabion.runtime import stable_encode


# gabion:evidence E:call_footprint::tests/test_order_lifetime_contract.py::test_sort_once_rejects_second_sort_on_boundary_carrier::order_contract.py::gabion.order_contract.sort_once::boundary_order.py::gabion.commands.boundary_order.canonicalize_boundary_mapping
def test_sort_once_rejects_second_sort_on_boundary_carrier() -> None:
    normalized = boundary_order.canonicalize_boundary_mapping(
        {"b": 2, "a": 1},
        source="tests.test_order_lifetime_contract.first_sort",
    )
    with pytest.raises(NeverThrown):
        sort_once(
            normalized,
            source="tests.test_order_lifetime_contract.second_sort",
        )


# gabion:evidence E:call_footprint::tests/test_order_lifetime_contract.py::test_enforce_ordered_rejects_unsorted_without_fallback::order_contract.py::gabion.order_contract.enforce_ordered
def test_enforce_ordered_rejects_unsorted_without_fallback() -> None:
    with pytest.raises(NeverThrown):
        enforce_ordered(
            ["z", "a"],
            source="tests.test_order_lifetime_contract.egress",
        )


# gabion:evidence E:call_footprint::tests/test_order_lifetime_contract.py::test_stable_encode_is_deterministic_for_mapping_order::stable_encode.py::gabion.runtime.stable_encode.stable_compact_text
def test_stable_encode_is_deterministic_for_mapping_order() -> None:
    first = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
    second = {"nested": {"x": 1, "y": 2}, "a": 1, "b": 2}
    assert stable_encode.stable_compact_text(first) == stable_encode.stable_compact_text(second)
