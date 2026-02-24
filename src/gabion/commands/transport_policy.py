from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Mapping, TypeAlias

from gabion.invariants import never
from gabion.lsp_client import run_command, run_command_direct
from gabion.tooling.governance_rules import CommandPolicy, load_governance_rules
from gabion.tooling.override_record import validate_override_record_json

Runner: TypeAlias = Callable[..., Mapping[str, object]]

DIRECT_RUN_ENV = "GABION_DIRECT_RUN"
DIRECT_RUN_OVERRIDE_EVIDENCE_ENV = "GABION_DIRECT_RUN_OVERRIDE_EVIDENCE"
OVERRIDE_RECORD_JSON_ENV = "GABION_OVERRIDE_RECORD_JSON"


@dataclass(frozen=True)
class CommandTransportDecision:
    runner: Runner
    direct_requested: bool
    direct_override_evidence: str | None
    direct_override_telemetry: Mapping[str, object] | None
    policy: CommandPolicy | None


# gabion:decision_protocol
def resolve_command_transport(*, command: str, runner: Runner) -> CommandTransportDecision:
    direct_flag = os.getenv(DIRECT_RUN_ENV, "").strip().lower()
    direct_requested = direct_flag in {"1", "true", "yes", "on"}
    override_evidence = os.getenv(DIRECT_RUN_OVERRIDE_EVIDENCE_ENV, "").strip() or None
    if runner is not run_command:
        return CommandTransportDecision(
            runner=runner,
            direct_requested=direct_requested,
            direct_override_evidence=override_evidence,
            direct_override_telemetry=None,
            policy=None,
        )

    policy = load_governance_rules().command_policies.get(command)
    require_lsp_carrier = False
    if policy is not None:
        require_lsp_carrier = policy.require_lsp_carrier or policy.maturity in {
            "beta",
            "production",
        }

    override_record = validate_override_record_json(os.getenv(OVERRIDE_RECORD_JSON_ENV))
    override_telemetry = None
    if direct_requested and require_lsp_carrier:
        if override_evidence is None:
            never(
                "direct transport forbidden by command maturity policy",
                command=command,
                maturity=(policy.maturity if policy is not None else "unknown"),
                override_evidence_env=DIRECT_RUN_OVERRIDE_EVIDENCE_ENV,
            )
        if not override_record.valid:
            never(
                "direct transport override record invalid",
                command=command,
                maturity=(policy.maturity if policy is not None else "unknown"),
                override_record_env=OVERRIDE_RECORD_JSON_ENV,
                **override_record.telemetry(source="command_transport"),
            )
        override_telemetry = override_record.telemetry(source="command_transport")
    resolved_runner = run_command_direct if direct_requested else run_command
    return CommandTransportDecision(
        runner=resolved_runner,
        direct_requested=direct_requested,
        direct_override_evidence=override_evidence,
        direct_override_telemetry=override_telemetry,
        policy=policy,
    )
