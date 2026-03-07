# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer

from gabion.commands import check_contract

CheckArtifactFlags = check_contract.CheckArtifactFlags
CheckDeltaOptions = check_contract.CheckDeltaOptions


def default_check_artifact_flags() -> CheckArtifactFlags:
    return CheckArtifactFlags(
        emit_test_obsolescence=False,
        emit_test_evidence_suggestions=False,
        emit_call_clusters=False,
        emit_call_cluster_consolidation=False,
        emit_test_annotation_drift=False,
        emit_semantic_coverage_map=False,
    )


def default_check_delta_options() -> CheckDeltaOptions:
    return CheckDeltaOptions(
        obsolescence_mode=check_contract.CheckAuxMode(kind="off"),
        annotation_drift_mode=check_contract.CheckAuxMode(kind="off"),
        ambiguity_mode=check_contract.CheckAuxMode(kind="off"),
        semantic_coverage_mapping=None,
    )


def check_help_or_exit(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=2)


def check_gate_policy(gate: object, *, never_fn: Callable[..., object]) -> tuple[bool, bool]:
    gate_value = str(getattr(gate, "value", gate))
    if gate_value == "all":
        return True, True
    if gate_value == "none":
        return False, False
    if gate_value == "violations":
        return True, False
    if gate_value == "type-ambiguities":
        return False, True
    never_fn("invalid check gate mode", gate=gate_value)
    return False, False  # pragma: no cover


def check_lint_mode(
    *,
    lint_mode: object,
    lint_jsonl_out: Path | None,
    lint_sarif_out: Path | None,
) -> tuple[bool, bool]:
    lint_value = str(getattr(lint_mode, "value", lint_mode))
    line_enabled = lint_value in {"line", "line+jsonl", "line+sarif", "all"}
    jsonl_enabled = lint_value in {"jsonl", "line+jsonl", "jsonl+sarif", "all"}
    sarif_enabled = lint_value in {"sarif", "line+sarif", "jsonl+sarif", "all"}
    if jsonl_enabled and lint_jsonl_out is None:
        raise typer.BadParameter(
            "--lint-jsonl-out is required when --lint includes jsonl output."
        )
    if sarif_enabled and lint_sarif_out is None:
        raise typer.BadParameter(
            "--lint-sarif-out is required when --lint includes sarif output."
        )
    if not jsonl_enabled and lint_jsonl_out is not None:
        raise typer.BadParameter(
            "--lint-jsonl-out is only valid when --lint includes jsonl."
        )
    if not sarif_enabled and lint_sarif_out is not None:
        raise typer.BadParameter(
            "--lint-sarif-out is only valid when --lint includes sarif."
        )
    lint_enabled = lint_value != "none"
    return lint_enabled, line_enabled
