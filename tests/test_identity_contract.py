from __future__ import annotations

from gabion.analysis.identity_contract import IdentityAxis, build_identity_contract


# gabion:evidence E:call_footprint::tests/test_identity_contract.py::test_build_identity_contract_is_stable_for_mapping_order::identity_contract.py::gabion.analysis.identity_contract.build_identity_contract
def test_build_identity_contract_is_stable_for_mapping_order() -> None:
    first = build_identity_contract(
        axis=IdentityAxis.SCHEMA,
        kind="execution_pattern",
        payload={
            "schema_contract": "pattern_schema.v2",
            "signature": {"b": ["z", "a"], "a": "x"},
        },
    )
    second = build_identity_contract(
        axis=IdentityAxis.SCHEMA,
        kind="execution_pattern",
        payload={
            "signature": {"a": "x", "b": ["a", "z"]},
            "schema_contract": "pattern_schema.v2",
        },
    )

    assert first.digest == second.digest
    assert first.canonical_payload == second.canonical_payload


# gabion:evidence E:call_footprint::tests/test_identity_contract.py::test_build_identity_contract_changes_when_kind_changes::identity_contract.py::gabion.analysis.identity_contract.build_identity_contract
def test_build_identity_contract_changes_when_kind_changes() -> None:
    base = build_identity_contract(
        axis=IdentityAxis.SCHEMA,
        kind="bundle_pattern",
        payload={"schema_contract": "pattern_schema.v2", "signature": {"k": "v"}},
    )
    changed = build_identity_contract(
        axis=IdentityAxis.SCHEMA,
        kind="execution_pattern",
        payload={"schema_contract": "pattern_schema.v2", "signature": {"k": "v"}},
    )

    assert base.digest != changed.digest
