from __future__ import annotations

from gabion.analysis.aspf_core import (
    AspfCanonicalIdentityContract,
    AspfOneCell,
    AspfTwoCellWitness,
    BasisZeroCell,
    SuiteSiteEndpoint,
    compose_1cells,
    identity_1cell,
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
from gabion.analysis.evidence_keys import fingerprint_identity_layers


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
    assert identity["derived"]["digest_alias"]["canonical"] is False
