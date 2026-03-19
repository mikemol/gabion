from __future__ import annotations

import pytest

from gabion import cli
from gabion.analysis.foundation import aspf_execution_fibration_impl
from gabion.analysis.surfaces import test_obsolescence_delta
from gabion.commands import progress_contract
from gabion.exceptions import NeverThrown
from gabion.runtime import coercion_contract
from gabion.server_core import coercion_contract as server_core_coercion_contract
from gabion.server_core import command_orchestrator
from gabion.server_core import command_orchestrator_primitives
from gabion.server_core import command_orchestrator_progress
from gabion.server_core import server_payload_dispatch
from gabion.runtime_shape_dispatch import (
    float_optional,
    int_optional,
    json_mapping_optional,
    str_optional,
)
from gabion.tooling.governance import normative_symdiff


class _UnknownRuntimeValue:
    pass


# gabion:behavior primary=desired
def test_runtime_shape_dispatch_preserves_bool_numeric_edges() -> None:
    assert json_mapping_optional({"ok": 1}) == {"ok": 1}
    assert json_mapping_optional(b"bytes") is None
    assert str_optional("value") == "value"
    assert str_optional(b"bytes") is None
    assert int_optional(7) == 7
    assert int_optional(True) is None
    assert float_optional(3) == 3.0
    assert float_optional(True) == 1.0


# gabion:behavior primary=desired
def test_shared_coercion_contract_preserves_progress_contract_policies() -> None:
    assert coercion_contract.MAPPING_OPTIONAL_POLICY({"phase": "post"}) == {"phase": "post"}
    assert coercion_contract.NON_BOOL_INT_OPTIONAL_POLICY(True) is None
    assert coercion_contract.INT_LIKE_OPTIONAL_POLICY(True) is True
    assert coercion_contract.NON_BOOL_FLOAT_OPTIONAL_POLICY(True) is None
    assert coercion_contract.ROW_FLOAT_OPTIONAL_POLICY(True) == 1.0
    assert progress_contract._str_key_dict_optional({1: "a", "two": 2}) == {"1": "a", "two": 2}
    assert coercion_contract.LIST_OPTIONAL_POLICY(["x"]) == ["x"]
    assert coercion_contract.LIST_OR_TUPLE_TO_LIST_OPTIONAL_POLICY(("x", "y")) == ["x", "y"]
    assert coercion_contract.FLOAT_ONLY_OPTIONAL_POLICY(1) is None
    assert coercion_contract.CORE_STR_OPTIONAL_POLICY("value") == "value"


# gabion:behavior primary=desired
def test_followon_modules_use_shared_coercion_substrate() -> None:
    assert cli._int_optional(complex(1, 2)) is None
    assert cli._str_optional(b"bytes") is None
    assert cli._mapping_optional(frozenset({"x"})) is None
    assert cli._json_object_optional({"ok": 1}) == {"ok": 1}
    assert command_orchestrator_progress._int_optional(True) is None
    assert command_orchestrator_progress._float_optional(1) is None
    assert command_orchestrator_progress._str_optional("phase") == "phase"
    assert normative_symdiff._list_optional(("a", "b")) == ["a", "b"]
    assert aspf_execution_fibration_impl._payload_mapping_optional({"ok": 1}) == {"ok": 1}
    assert test_obsolescence_delta._list_optional(["item"]) == ["item"]
    assert test_obsolescence_delta._str_optional("item") == "item"


# gabion:behavior primary=desired
def test_server_core_coercion_contract_preserves_orchestrator_edges() -> None:
    assert server_core_coercion_contract.object_mapping_optional({"ok": 1}) == {"ok": 1}
    assert server_core_coercion_contract.string_optional("phase") == "phase"
    assert server_core_coercion_contract.bool_optional(True) is True
    assert server_core_coercion_contract.bool_optional(1) is None
    assert server_core_coercion_contract.non_negative_float_optional(True) == 1.0
    assert server_core_coercion_contract.non_negative_float_optional(-2.0) == 0.0
    assert server_core_coercion_contract.int_or_zero(True) == 1
    assert server_core_coercion_contract.int_or_zero("nope") == 0
    assert server_core_coercion_contract.non_string_sequence_optional(("a", "b")) == ("a", "b")
    assert server_core_coercion_contract.non_string_sequence_optional({"a", "b"}) == {"a", "b"}
    assert server_core_coercion_contract.float_optional(1) is None
    assert server_core_coercion_contract.json_mapping_default_empty(None) == {}


# gabion:behavior primary=desired
def test_server_core_modules_bind_to_shared_coercion_contract() -> None:
    assert cli._int_optional is server_core_coercion_contract.cli_int_optional
    assert cli._str_optional is server_core_coercion_contract.cli_str_optional
    assert cli._mapping_optional is server_core_coercion_contract.cli_mapping_optional
    assert (
        cli._json_object_optional
        is server_core_coercion_contract.cli_json_object_optional
    )
    assert server_core_coercion_contract.bool_optional is server_core_coercion_contract._bool_optional
    assert command_orchestrator._bool_optional is server_core_coercion_contract.bool_optional
    assert (
        command_orchestrator._non_negative_float_optional
        is server_core_coercion_contract.non_negative_float_optional
    )
    assert command_orchestrator._int_or_zero is server_core_coercion_contract.int_or_zero
    assert command_orchestrator_progress._bool_optional is server_core_coercion_contract.bool_optional
    assert (
        command_orchestrator_progress._non_string_sequence_optional
        is server_core_coercion_contract.non_string_sequence_optional
    )
    assert command_orchestrator_progress._float_optional is server_core_coercion_contract.float_optional
    assert command_orchestrator_primitives._bool_optional is server_core_coercion_contract.bool_optional
    assert (
        command_orchestrator_primitives._json_mapping_optional
        is server_core_coercion_contract.json_mapping_optional
    )
    assert server_payload_dispatch._bool_optional is server_core_coercion_contract.bool_optional
    assert (
        server_payload_dispatch._non_string_sequence_optional
        is server_core_coercion_contract.non_string_sequence_optional
    )


# gabion:behavior primary=desired
def test_coercion_contract_rejects_unregistered_runtime_types() -> None:
    with pytest.raises(NeverThrown, match="unregistered runtime type"):
        coercion_contract.STR_OPTIONAL_POLICY(_UnknownRuntimeValue())

    with pytest.raises(NeverThrown, match="unregistered runtime type"):
        progress_contract._mapping_optional(_UnknownRuntimeValue())

    with pytest.raises(NeverThrown, match="unregistered runtime type"):
        server_core_coercion_contract._bool_optional(_UnknownRuntimeValue())
