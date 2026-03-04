# gabion:decision_protocol_module
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Protocol, Sequence

from gabion.cli_support.check.check_execution_plan import (
    build_check_execution_plan_request as _build_check_execution_plan_request_impl,
    check_derived_artifacts as _check_derived_artifacts_impl,
)
from gabion.cli_support.check.check_runtime import run_check as _run_check_impl
from gabion.cli_support.shared.dispatch_runtime import (
    dispatch_command as _dispatch_command_impl,
)
from gabion.cli_support.shared import dataflow_runtime_common
from gabion.cli_support.shared.payload_builder import (
    build_dataflow_payload as _build_dataflow_payload_impl,
)
from gabion.cli_support.shared.raw_argparse import (
    parse_dataflow_args_or_exit as _parse_dataflow_args_or_exit_impl,
)
from gabion.commands import boundary_order, check_contract, command_ids, transport_policy
from gabion.commands.check_contract import DataflowFilterBundle
from gabion.invariants import never
from gabion.json_types import JSONObject
from gabion.lsp_client import CommandRequest, run_command, run_command_direct
from gabion.tooling.runtime.execution_envelope import ExecutionEnvelope

_STDOUT_ALIAS = "-"
_STDOUT_PATH = "/dev/stdout"


class _ExecutionPlanRequest(Protocol):
    def to_payload(self) -> JSONObject: ...


@dataclass(frozen=True)
class _ExecutionPlanRequestPayload:
    requested_operations: list[str]
    inputs: JSONObject
    derived_artifacts: list[str]
    obligations: dict[str, list[str]]
    policy_metadata: dict[str, object]

    def to_payload(self) -> JSONObject:
        return {
            "requested_operations": list(self.requested_operations),
            "inputs": dict(self.inputs),
            "derived_artifacts": list(self.derived_artifacts),
            "obligations": {
                "preconditions": list(self.obligations.get("preconditions") or []),
                "postconditions": list(self.obligations.get("postconditions") or []),
            },
            "policy_metadata": dict(self.policy_metadata),
        }


def _cli_timeout_ticks() -> tuple[int, int]:
    return dataflow_runtime_common.cli_timeout_ticks(
        default_ticks=dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICKS,
        default_tick_ns=dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICK_NS,
    )


def _resolve_check_report_path(report: Path | None, *, root: Path) -> Path:
    return dataflow_runtime_common.resolve_check_report_path(report, root=root)


def _normalize_output_target(target: str | Path) -> str:
    return dataflow_runtime_common.normalize_output_target(
        target,
        stdout_alias=_STDOUT_ALIAS,
        stdout_path=_STDOUT_PATH,
    )


def _normalize_optional_output_target(target: object) -> str | None:
    return dataflow_runtime_common.normalize_optional_output_target(
        target,
        stdout_alias=_STDOUT_ALIAS,
        stdout_path=_STDOUT_PATH,
    )


def _build_dataflow_payload_common(
    *,
    options: check_contract.DataflowPayloadCommonOptions,
) -> JSONObject:
    return dataflow_runtime_common.build_dataflow_payload_common(options=options)


def _build_dataflow_payload(opts: argparse.Namespace) -> JSONObject:
    return _build_dataflow_payload_impl(
        opts,
        normalize_optional_output_target_fn=_normalize_optional_output_target,
        build_dataflow_payload_common_fn=_build_dataflow_payload_common,
    )


def _build_check_execution_plan_request(**kwargs) -> _ExecutionPlanRequest:
    return _build_check_execution_plan_request_impl(
        **kwargs,
        check_derived_artifacts_fn=_check_derived_artifacts_impl,
        execution_plan_request_ctor=_ExecutionPlanRequestPayload,
        dataflow_command=command_ids.DATAFLOW_COMMAND,
        check_command=command_ids.CHECK_COMMAND,
    )


def _dispatch_command(
    *,
    command: str,
    payload: JSONObject,
    root: Path = Path("."),
    runner: Callable[..., JSONObject] = run_command,
    execution_plan_request: _ExecutionPlanRequest | None = None,
) -> JSONObject:
    return _dispatch_command_impl(
        command=command,
        payload=payload,
        root=root,
        runner=runner,
        execution_plan_request=execution_plan_request,
        notification_callback=None,
        cli_timeout_ticks_fn=_cli_timeout_ticks,
        normalize_boundary_mapping_once_fn=boundary_order.normalize_boundary_mapping_once,
        apply_boundary_updates_once_fn=boundary_order.apply_boundary_updates_once,
        enforce_boundary_mapping_ordered_fn=boundary_order.enforce_boundary_mapping_ordered,
        command_request_ctor=CommandRequest,
        resolve_command_transport_fn=transport_policy.resolve_command_transport,
        default_lsp_runner=run_command,
        direct_runner=run_command_direct,
        never_fn=never,
    )


def _run_check(**kwargs) -> JSONObject:
    return _run_check_impl(
        **kwargs,
        runner=run_command,
        resolve_check_report_path_fn=_resolve_check_report_path,
        build_check_payload_fn=check_contract.build_check_payload,
        build_check_execution_plan_request_fn=_build_check_execution_plan_request,
        dispatch_command_fn=_dispatch_command,
        dataflow_command=command_ids.DATAFLOW_COMMAND,
    )


@dataclass(frozen=True)
class DataflowInvocationResult:
    exit_code: int
    analysis_state: str
    payload: JSONObject


@dataclass(frozen=True)
class DataflowInvocationRunner:
    run_check_fn: Callable[..., JSONObject] | None = None
    dispatch_command_fn: Callable[..., JSONObject] | None = None
    parse_dataflow_args_fn: Callable[[list[str]], argparse.Namespace] | None = None
    build_dataflow_payload_fn: Callable[[argparse.Namespace], JSONObject] | None = None
    run_command_fn: Callable[..., JSONObject] | None = None

    def _resolve_run_check(self) -> Callable[..., JSONObject]:
        if callable(self.run_check_fn):
            return self.run_check_fn
        return _run_check

    def _resolve_dispatch_command(self) -> Callable[..., JSONObject]:
        if callable(self.dispatch_command_fn):
            return self.dispatch_command_fn
        return _dispatch_command

    def _resolve_parse_dataflow_args(self) -> Callable[[list[str]], argparse.Namespace]:
        if callable(self.parse_dataflow_args_fn):
            return self.parse_dataflow_args_fn
        return _parse_dataflow_args_or_exit_impl

    def _resolve_build_dataflow_payload(
        self,
    ) -> Callable[[argparse.Namespace], JSONObject]:
        if callable(self.build_dataflow_payload_fn):
            return self.build_dataflow_payload_fn
        return _build_dataflow_payload

    def _resolve_run_command(self) -> Callable[..., JSONObject]:
        if callable(self.run_command_fn):
            return self.run_command_fn
        return run_command

    def _ensure_repo_root_importable(self, root: Path) -> None:
        root_text = str(root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

    # gabion:decision_protocol
    def run_delta_bundle(
        self,
        envelope: ExecutionEnvelope,
    ) -> DataflowInvocationResult:
        checked = envelope.validate()
        if checked.operation != "delta_bundle":
            raise ValueError("execution envelope operation must be 'delta_bundle'")
        self._ensure_repo_root_importable(checked.root)
        run_check = self._resolve_run_check()
        payload = run_check(
            paths=None,
            report=checked.report_path,
            policy=check_contract.CheckPolicyFlags(
                fail_on_violations=False,
                fail_on_type_ambiguities=False,
                lint=False,
            ),
            root=checked.root,
            config=None,
            baseline=None,
            baseline_write=False,
            decision_snapshot=None,
            artifact_flags=check_contract.delta_bundle_artifact_flags(),
            delta_options=check_contract.delta_bundle_delta_options(),
            exclude=None,
            filter_bundle=DataflowFilterBundle(
                ignore_params_csv=None,
                transparent_decorators_csv=None,
            ),
            allow_external=checked.allow_external,
            strictness=checked.strictness,
            analysis_tick_limit=None,
            aspf_trace_json=None,
            aspf_import_trace=None,
            aspf_equivalence_against=None,
            aspf_opportunities_json=None,
            aspf_state_json=checked.aspf_state_json,
            aspf_import_state=list(checked.aspf_import_state),
            aspf_delta_jsonl=checked.aspf_delta_jsonl,
            aspf_semantic_surface=None,
        )
        return DataflowInvocationResult(
            exit_code=int(payload.get("exit_code", 0) or 0),
            analysis_state=str(payload.get("analysis_state", "") or "none"),
            payload=payload,
        )

    # gabion:decision_protocol
    def run_raw(
        self,
        envelope: ExecutionEnvelope,
        raw_args: Sequence[str],
    ) -> DataflowInvocationResult:
        checked = envelope.validate()
        if checked.operation != "raw":
            raise ValueError("execution envelope operation must be 'raw'")
        self._ensure_repo_root_importable(checked.root)
        parse_dataflow_args_or_exit = self._resolve_parse_dataflow_args()
        build_dataflow_payload = self._resolve_build_dataflow_payload()
        run_command_fn = self._resolve_run_command()
        dispatch_command = self._resolve_dispatch_command()
        opts = parse_dataflow_args_or_exit(list(raw_args))
        payload = build_dataflow_payload(opts)
        if checked.aspf_state_json is not None:
            payload["aspf_state_json"] = str(checked.aspf_state_json)
            payload["aspf_delta_jsonl"] = str(checked.aspf_delta_jsonl)
            payload["aspf_import_state"] = [
                str(path) for path in checked.aspf_import_state
            ]
        result = dispatch_command(
            command=command_ids.DATAFLOW_COMMAND,
            payload=payload,
            root=Path(opts.root),
            runner=run_command_fn,
        )
        return DataflowInvocationResult(
            exit_code=int(result.get("exit_code", 0) or 0),
            analysis_state=str(result.get("analysis_state", "") or "none"),
            payload=result,
        )


__all__ = [
    "DataflowInvocationResult",
    "DataflowInvocationRunner",
]
