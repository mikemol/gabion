from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion.commands import aux_operation_contract
from gabion.exceptions import NeverThrown


def test_evaluate_aux_operation_normalizes_domain_and_action() -> None:
    decision = aux_operation_contract.evaluate_aux_operation(
        domain="  Obsolescence ",
        action="  Report ",
        baseline_path=None,
    )
    assert decision.domain == "obsolescence"
    assert decision.action == "report"
    assert decision.baseline_path_required is False


@pytest.mark.parametrize(
    ("domain", "action", "baseline_path"),
    [
        ("unknown", "state", None),
        ("ambiguity", "report", None),
        ("taint", "delta", None),
        ("obsolescence", "baseline-write", None),
    ],
)
def test_validate_aux_operation_for_typer_rejects_invalid_combinations(
    domain: object,
    action: object,
    baseline_path: Path | None,
) -> None:
    with pytest.raises(typer.BadParameter):
        aux_operation_contract.validate_aux_operation_for_typer(
            domain=domain,
            action=action,
            baseline_path=baseline_path,
        )


@pytest.mark.parametrize(
    ("domain", "action", "baseline_path"),
    [
        ("unknown", "state", None),
        ("ambiguity", "report", None),
        ("taint", "delta", None),
    ],
)
def test_validate_aux_operation_or_never_rejects_invalid_combinations(
    domain: object,
    action: object,
    baseline_path: Path | None,
) -> None:
    with pytest.raises(NeverThrown):
        aux_operation_contract.validate_aux_operation_or_never(
            domain=domain,
            action=action,
            baseline_path=baseline_path,
        )


def test_aux_operation_contract_error_never_details_optional_fields() -> None:
    error = aux_operation_contract.AuxOperationContractError(
        kind="invalid_domain",
        domain="x",
        action="y",
        allowed_domains=(),
        allowed_actions=(),
    )
    details = error.never_details()
    assert details == {"domain": "x", "action": "y"}
