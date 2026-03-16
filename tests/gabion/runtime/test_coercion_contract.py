from __future__ import annotations

import pytest

from gabion.commands import progress_contract
from gabion.exceptions import NeverThrown
from gabion.runtime import coercion_contract
from gabion.runtime_shape_dispatch import (
    float_optional,
    int_optional,
    json_mapping_optional,
    str_optional,
)


class _UnknownRuntimeValue:
    pass


def test_runtime_shape_dispatch_preserves_bool_numeric_edges() -> None:
    assert json_mapping_optional({"ok": 1}) == {"ok": 1}
    assert json_mapping_optional(b"bytes") is None
    assert str_optional("value") == "value"
    assert str_optional(b"bytes") is None
    assert int_optional(7) == 7
    assert int_optional(True) is None
    assert float_optional(3) == 3.0
    assert float_optional(True) == 1.0


def test_shared_coercion_contract_preserves_progress_contract_policies() -> None:
    assert coercion_contract.MAPPING_OPTIONAL_POLICY({"phase": "post"}) == {"phase": "post"}
    assert coercion_contract.NON_BOOL_INT_OPTIONAL_POLICY(True) is None
    assert coercion_contract.INT_LIKE_OPTIONAL_POLICY(True) is True
    assert coercion_contract.NON_BOOL_FLOAT_OPTIONAL_POLICY(True) is None
    assert coercion_contract.ROW_FLOAT_OPTIONAL_POLICY(True) == 1.0
    assert progress_contract._str_key_dict_optional({1: "a", "two": 2}) == {"1": "a", "two": 2}


def test_coercion_contract_rejects_unregistered_runtime_types() -> None:
    with pytest.raises(NeverThrown, match="unregistered runtime type"):
        coercion_contract.STR_OPTIONAL_POLICY(_UnknownRuntimeValue())

    with pytest.raises(NeverThrown, match="unregistered runtime type"):
        progress_contract._mapping_optional(_UnknownRuntimeValue())
