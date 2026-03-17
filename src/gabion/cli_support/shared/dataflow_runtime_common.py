from __future__ import annotations

from pathlib import Path

from gabion.commands import check_contract
from gabion.json_types import JSONObject
from gabion.runtime import deadline_policy, path_policy

DEFAULT_CLI_TIMEOUT_TICKS = 7_500
DEFAULT_CLI_TIMEOUT_TICK_NS = 1_000_000
STDOUT_ALIAS = "-"
STDOUT_PATH = "/dev/stdout"


def cli_timeout_ticks(
    *,
    default_ticks: int = DEFAULT_CLI_TIMEOUT_TICKS,
    default_tick_ns: int = DEFAULT_CLI_TIMEOUT_TICK_NS,
) -> tuple[int, int]:
    budget = deadline_policy.timeout_budget_from_lsp_env(
        default_budget=deadline_policy.DeadlineBudget(
            ticks=default_ticks,
            tick_ns=default_tick_ns,
        )
    )
    return budget.ticks, budget.tick_ns


def resolve_check_report_path(report: Path | None, *, root: Path) -> Path:
    return path_policy.resolve_report_path(report, root=root)


def normalize_output_target(
    target: str | Path,
    *,
    stdout_alias: str = STDOUT_ALIAS,
    stdout_path: str = STDOUT_PATH,
) -> str:
    return path_policy.normalize_output_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def normalize_optional_output_target(
    target: object,
    *,
    stdout_alias: str = STDOUT_ALIAS,
    stdout_path: str = STDOUT_PATH,
) -> str | None:
    if target is None:
        return None
    text = str(target).strip()
    if not text:
        return None
    return normalize_output_target(
        text,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def build_dataflow_payload_common(
    *,
    options: check_contract.DataflowPayloadCommonOptions,
) -> JSONObject:
    return check_contract.build_dataflow_payload_common(options=options)

