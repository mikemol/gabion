# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from gabion.invariants import never

AUX_OPERATION_DOMAIN_ACTIONS: dict[str, tuple[str, ...]] = {
    "obsolescence": ("report", "state", "delta", "baseline-write"),
    "annotation-drift": ("report", "state", "delta", "baseline-write"),
    "ambiguity": ("state", "delta", "baseline-write"),
    "taint": ("state", "delta", "baseline-write", "lifecycle"),
}
AUX_OPERATION_BASELINE_REQUIRED_ACTIONS: tuple[str, ...] = ("delta", "baseline-write")


@dataclass(frozen=True)
class AuxOperationDecision:
    domain: str
    action: str
    baseline_path_required: bool
    allowed_actions: tuple[str, ...]


@dataclass(frozen=True)
class AuxOperationContractError(Exception):
    kind: Literal[
        "invalid_domain",
        "invalid_action",
        "missing_baseline_path",
    ]
    domain: str
    action: str
    allowed_domains: tuple[str, ...]
    allowed_actions: tuple[str, ...]

    def bad_parameter_message(self) -> str:
        if self.kind == "invalid_domain":
            return (
                "aux_operation domain must be one of: "
                "obsolescence, annotation-drift, ambiguity, taint."
            )
        if self.kind == "invalid_action":
            return (
                f"aux_operation action '{self.action}' "
                f"is not valid for domain '{self.domain}'."
            )
        return (
            "aux_operation requires baseline_path for "
            "delta and baseline-write actions."
        )

    def never_message(self) -> str:
        if self.kind == "invalid_domain":
            return "invalid aux operation domain"
        if self.kind == "invalid_action":
            return "invalid aux operation action"
        return "aux operation missing baseline path"

    def never_details(self) -> dict[str, object]:
        details: dict[str, object] = {
            "domain": self.domain,
            "action": self.action,
        }
        if self.allowed_domains:
            details["allowed_domains"] = sorted(self.allowed_domains)
        if self.allowed_actions:
            details["allowed_actions"] = sorted(self.allowed_actions)
        return details


def normalize_aux_operation_domain(value: object) -> str:
    return str(value or "").strip().lower()


def normalize_aux_operation_action(value: object) -> str:
    return str(value or "").strip().lower()


def evaluate_aux_operation(
    *,
    domain: object,
    action: object,
    baseline_path: Path | None,
) -> AuxOperationDecision:
    normalized_domain = normalize_aux_operation_domain(domain)
    normalized_action = normalize_aux_operation_action(action)
    allowed_domains = tuple(sorted(AUX_OPERATION_DOMAIN_ACTIONS))
    allowed_actions = AUX_OPERATION_DOMAIN_ACTIONS.get(normalized_domain)
    if allowed_actions is None:
        raise AuxOperationContractError(
            kind="invalid_domain",
            domain=normalized_domain,
            action=normalized_action,
            allowed_domains=allowed_domains,
            allowed_actions=(),
        )
    if normalized_action not in allowed_actions:
        raise AuxOperationContractError(
            kind="invalid_action",
            domain=normalized_domain,
            action=normalized_action,
            allowed_domains=allowed_domains,
            allowed_actions=allowed_actions,
        )
    baseline_path_required = normalized_action in AUX_OPERATION_BASELINE_REQUIRED_ACTIONS
    if baseline_path_required and baseline_path is None:
        raise AuxOperationContractError(
            kind="missing_baseline_path",
            domain=normalized_domain,
            action=normalized_action,
            allowed_domains=allowed_domains,
            allowed_actions=allowed_actions,
        )
    return AuxOperationDecision(
        domain=normalized_domain,
        action=normalized_action,
        baseline_path_required=baseline_path_required,
        allowed_actions=allowed_actions,
    )


def validate_aux_operation_for_typer(
    *,
    domain: object,
    action: object,
    baseline_path: Path | None,
) -> AuxOperationDecision:
    try:
        return evaluate_aux_operation(
            domain=domain,
            action=action,
            baseline_path=baseline_path,
        )
    except AuxOperationContractError as exc:
        raise typer.BadParameter(exc.bad_parameter_message()) from None


def validate_aux_operation_or_never(
    *,
    domain: object,
    action: object,
    baseline_path: Path | None,
) -> AuxOperationDecision:
    try:
        return evaluate_aux_operation(
            domain=domain,
            action=action,
            baseline_path=baseline_path,
        )
    except AuxOperationContractError as exc:
        never(exc.never_message(), **exc.never_details())
    return AuxOperationDecision(
        domain="",
        action="",
        baseline_path_required=False,
        allowed_actions=(),
    )  # pragma: no cover - never() raises
