from __future__ import annotations

import pytest
from typing import cast

from gabion.analysis.aspf_core import (
    AspfCanonicalIdentityContract,
    AspfOneCell,
    AspfTwoCellWitness,
    BasisZeroCell,
    SuiteSiteEndpoint,
    compose_1cells,
    identity_1cell,
    parse_2cell_witness,
    validate_2cell_compatibility,
)
from gabion.analysis.aspf_decision_surface import (
    RepresentativeSelectionMode,
    RepresentativeSelectionOptions,
    classify_drift_by_homotopy,
    select_representative,
)
from gabion.analysis.aspf_morphisms import (
    AspfPrimeBasis,
    CofibrationWitnessCarrier,
    DomainPrimeBasis,
    DomainToAspfCofibration,
    DomainToAspfCofibrationEntry,
)
from gabion.analysis.evidence_keys import (
    CanonicalAspfPathPayload,
    DerivedIdentityAdapterLifecycle,
    DerivedIdentityProjection,
    FingerprintIdentityLayers,
    fingerprint_identity_layers,
)
from gabion.exceptions import NeverThrown


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_aspf_identity_and_associativity
def test_aspf_identity_and_associativity() -> None:
    a = BasisZeroCell("A")
    b = BasisZeroCell("B")
    c = BasisZeroCell("C")
    ab = AspfOneCell(a, b, "ab", ("A", "B"))
    bc = AspfOneCell(b, c, "bc", ("B", "C"))

    left = compose_1cells(identity_1cell(a), ab)
    right = compose_1cells(ab, identity_1cell(b))
    assert left.basis_path == ab.basis_path
    assert right.basis_path == ab.basis_path

    composed = compose_1cells(ab, bc)
    assert composed.basis_path == ("A", "B", "C")


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_higher_path_equivalence_and_drift_quotienting
def test_higher_path_equivalence_and_drift_quotienting() -> None:
    a = BasisZeroCell("A")
    b = BasisZeroCell("B")
    left = AspfOneCell(a, b, "left", ("A", "B"))
    right = AspfOneCell(a, b, "right", ("A", "B"))
    witness = AspfTwoCellWitness(left=left, right=right, witness_id="w:1", reason="equiv")
    validate_2cell_compatibility(witness)

    assert classify_drift_by_homotopy(
        baseline_representative="left",
        current_representative="right",
        equivalence_witness=witness,
    ) == "non_drift"
    assert classify_drift_by_homotopy(
        baseline_representative="left",
        current_representative="right",
        has_equivalence_witness=False,
    ) == "drift"


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_cofibration_injective_and_faithful
def test_cofibration_injective_and_faithful() -> None:
    cofibration = DomainToAspfCofibration(
        entries=(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:int", 2),
                aspf=AspfPrimeBasis("a:int", 2),
            ),
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:str", 3),
                aspf=AspfPrimeBasis("a:str", 3),
            ),
        )
    )
    cofibration.validate()
    carrier = CofibrationWitnessCarrier(
        canonical_identity_kind="canonical_aspf_structural_identity",
        cofibration=cofibration,
    )
    assert carrier.as_dict()["cofibration"]["entries"]

# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_canonical_identity_contract_carries_suite_site_endpoints
def test_canonical_identity_contract_carries_suite_site_endpoints() -> None:
    contract = AspfCanonicalIdentityContract(
        identity_kind="canonical_aspf_structural_identity",
        source=BasisZeroCell("fingerprint:start"),
        target=BasisZeroCell("fingerprint:end"),
        representative="rep",
        basis_path=("a", "b"),
        suite_site_source=SuiteSiteEndpoint("SuiteSite", ("fingerprint:start", "fingerprint", "source")),
        suite_site_target=SuiteSiteEndpoint("SuiteSite", ("fingerprint:end", "fingerprint", "target")),
    )
    payload = contract.as_dict()
    assert payload["suite_site_endpoints"]["source"]["kind"] == "SuiteSite"
    assert payload["suite_site_endpoints"]["target"]["key"] == [
        "fingerprint:end",
        "fingerprint",
        "target",
    ]
# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_deterministic_representative_selection_and_identity_layers
def test_deterministic_representative_selection_and_identity_layers() -> None:
    witness = select_representative(
        RepresentativeSelectionOptions(
            mode=RepresentativeSelectionMode.SHORTEST_PATH_THEN_LEXICOGRAPHIC,
            candidates=("zz", "a", "bbb"),
        )
    )
    assert witness.selected == "a"

    identity = fingerprint_identity_layers(
        canonical_aspf_path={"representative": witness.selected, "basis_path": ["a"]},
        scalar_prime_product=2,
    ).as_dict()
    assert identity["identity_layer"] == "canonical_aspf_path"
    assert identity["derived"]["scalar_prime_product"]["canonical"] is False
    assert identity["derived"]["scalar_prime_product"]["adapter_lifecycle"]["adapter_name"] == "scalar_prime_product"
    assert identity["derived"]["digest_alias"]["canonical"] is False


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_aspf_edge_guards_and_parse_paths
def test_aspf_edge_guards_and_parse_paths() -> None:
    a = BasisZeroCell("A")
    b = BasisZeroCell("B")
    c = BasisZeroCell("C")
    ab = AspfOneCell(a, b, "ab", ("A", "B"))
    ca = AspfOneCell(c, a, "ca", ("C", "A"))

    with pytest.raises(ValueError):
        compose_1cells(ab, ca)

    incompatible = AspfTwoCellWitness(
        left=AspfOneCell(a, b, "left", ("A", "B")),
        right=AspfOneCell(a, b, "right", ("A", "C")),
        witness_id="w:bad",
        reason="bad",
    )
    with pytest.raises(ValueError):
        validate_2cell_compatibility(incompatible)

    assert parse_2cell_witness({"left": "bad", "right": {}}) is None
    assert parse_2cell_witness({"left": {}, "right": {}, "witness_id": 1, "reason": "r"}) is None
    assert parse_2cell_witness(
        {
            "left": {"source": "A", "target": "B", "representative": "ab", "basis_path": "bad"},
            "right": {"source": "A", "target": "B", "representative": "ab", "basis_path": ["A", "B"]},
            "witness_id": "w:1",
            "reason": "ok",
        }
    ) is None
    assert parse_2cell_witness(
        {
            "left": {"source": 1, "target": "B", "representative": "ab", "basis_path": ["A", "B"]},
            "right": {"source": "A", "target": "B", "representative": "ab", "basis_path": ["A", "B"]},
            "witness_id": "w:1",
            "reason": "ok",
        }
    ) is None
    assert parse_2cell_witness(
        {
            "left": {"source": "A", "target": "B", "representative": "ab", "basis_path": ["A", "B"]},
            "right": {"source": "A", "target": "B", "representative": "ab", "basis_path": ["A", "B"]},
            "witness_id": "w:1",
            "reason": "ok",
        }
    ) is not None


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_selection_and_cofibration_failure_edges
def test_selection_and_cofibration_failure_edges() -> None:
    with pytest.raises(ValueError):
        RepresentativeSelectionOptions(
            mode=RepresentativeSelectionMode.LEXICOGRAPHIC_MIN,
            candidates=(),
        ).validate()
    with pytest.raises(ValueError):
        RepresentativeSelectionOptions(
            mode=RepresentativeSelectionMode.LEXICOGRAPHIC_MIN,
            candidates=("dup", "dup"),
        ).validate()

    with pytest.raises(NeverThrown, match="unsupported representative selection mode"):
        select_representative(
            RepresentativeSelectionOptions(
                mode=cast(RepresentativeSelectionMode, "unsupported_mode"),
                candidates=("candidate",),
            )
        )
    assert (
        classify_drift_by_homotopy(
            baseline_representative="left",
            current_representative="right",
            has_equivalence_witness=True,
        )
        == "non_drift"
    )
    mismatch_witness = AspfTwoCellWitness(
        left=AspfOneCell(BasisZeroCell("L"), BasisZeroCell("R"), "left", ("L", "R")),
        right=AspfOneCell(BasisZeroCell("L"), BasisZeroCell("R"), "other", ("L", "R")),
        witness_id="w:mismatch",
        reason="mismatch",
    )
    assert (
        classify_drift_by_homotopy(
            baseline_representative="left",
            current_representative="right",
            equivalence_witness=mismatch_witness,
            has_equivalence_witness=False,
        )
        == "drift"
    )

    with pytest.raises(ValueError):
        DomainToAspfCofibration(entries=()).validate()

    duplicate_target = DomainToAspfCofibration(
        entries=(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:int", 2),
                aspf=AspfPrimeBasis("a:int", 2),
            ),
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:str", 3),
                aspf=AspfPrimeBasis("a:int", 3),
            ),
        )
    )
    with pytest.raises(ValueError):
        duplicate_target.validate()

    non_prime = DomainToAspfCofibration(
        entries=(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:int", 1),
                aspf=AspfPrimeBasis("a:int", 1),
            ),
        )
    )
    with pytest.raises(ValueError):
        non_prime.validate()

    non_faithful = DomainToAspfCofibration(
        entries=(
            DomainToAspfCofibrationEntry(
                domain=DomainPrimeBasis("d:int", 2),
                aspf=AspfPrimeBasis("a:int", 3),
            ),
        )
    )
    with pytest.raises(ValueError):
        non_faithful.validate()


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_identity_layer_projection_contract_guards
def test_identity_layer_projection_contract_guards() -> None:
    broken = FingerprintIdentityLayers(
        canonical=CanonicalAspfPathPayload(representative="a", basis_path=("a",)),
        scalar_projection=DerivedIdentityProjection(
            value=2,
            canonical=True,
            projection="prime_product",
            adapter_lifecycle=DerivedIdentityAdapterLifecycle(
                actor="a",
                rationale="r",
                scope="s",
                start="st",
                expiry="ex",
                rollback_condition="rb",
                evidence_links=("e",),
                adapter_name="scalar_prime_product",
            ),
        ),
        digest_projection=DerivedIdentityProjection(
            value="d",
            canonical=False,
            projection="digest_alias",
            adapter_lifecycle=DerivedIdentityAdapterLifecycle(
                actor="a",
                rationale="r",
                scope="s",
                start="st",
                expiry="ex",
                rollback_condition="rb",
                evidence_links=("e",),
                adapter_name="digest_alias",
            ),
        ),
    )
    with pytest.raises(NeverThrown):
        broken.validate()
    with pytest.raises(NeverThrown):
        broken.as_dict()
    with pytest.raises(NeverThrown):
        broken.derived_aliases(alias_of="canonical_identity_contract")


# gabion:evidence E:function_site::tests/test_aspf_cofibration_laws.py::tests.test_aspf_cofibration_laws.test_fingerprint_identity_layers_guard_invalid_canonical_payload
def test_fingerprint_identity_layers_guard_invalid_canonical_payload() -> None:
    with pytest.raises(NeverThrown, match="canonical identity representative must be non-empty"):
        fingerprint_identity_layers(
            canonical_aspf_path={"representative": "", "basis_path": ["a"]},
            scalar_prime_product=2,
        )

    with pytest.raises(NeverThrown, match="canonical identity suite_site endpoint kind must be non-empty"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "", "key": ["left"]},
                    "target": {"kind": "SuiteSite", "key": ["right"]},
                },
            },
            scalar_prime_product=2,
        )

    with pytest.raises(NeverThrown, match="canonical identity suite_site endpoint kind must be SuiteSite"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "OtherSite", "key": ["left", "fingerprint", "source"]},
                    "target": {"kind": "SuiteSite", "key": ["right", "fingerprint", "target"]},
                },
            },
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity suite_site endpoint key must contain at least one segment"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "SuiteSite", "key": []},
                    "target": {"kind": "SuiteSite", "key": ["right"]},
                },
            },
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity basis_path must contain at least one segment"):
        fingerprint_identity_layers(
            canonical_aspf_path={"representative": "rep", "basis_path": []},
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity source/target must be both present or both empty"):
        fingerprint_identity_layers(
            canonical_aspf_path={"representative": "rep", "basis_path": ["a"], "source": "left"},
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity suite_site endpoint must be a mapping"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {"source": {"kind": "SuiteSite", "key": ["left"]}},
            },
            scalar_prime_product=2,
        )

    with pytest.raises(NeverThrown, match="canonical identity suite_site source key must start with source label"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "SuiteSite", "key": ["wrong", "fingerprint", "source"]},
                    "target": {"kind": "SuiteSite", "key": ["right", "fingerprint", "target"]},
                },
            },
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity suite_site target key must start with target label"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "SuiteSite", "key": ["left", "fingerprint", "source"]},
                    "target": {"kind": "SuiteSite", "key": ["wrong", "fingerprint", "target"]},
                },
            },
            scalar_prime_product=2,
        )

    with pytest.raises(NeverThrown, match="canonical identity suite_site source key must end with source role"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "SuiteSite", "key": ["left", "fingerprint", "wrong"]},
                    "target": {"kind": "SuiteSite", "key": ["right", "fingerprint", "target"]},
                },
            },
            scalar_prime_product=2,
        )
    with pytest.raises(NeverThrown, match="canonical identity suite_site target key must end with target role"):
        fingerprint_identity_layers(
            canonical_aspf_path={
                "representative": "rep",
                "basis_path": ["a"],
                "source": "left",
                "target": "right",
                "suite_site_endpoints": {
                    "source": {"kind": "SuiteSite", "key": ["left", "fingerprint", "source"]},
                    "target": {"kind": "SuiteSite", "key": ["right", "fingerprint", "wrong"]},
                },
            },
            scalar_prime_product=2,
        )
