from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gabion.cli_support.check.check_execution_plan import (
    build_check_execution_plan_request as _build_check_execution_plan_request_impl,
    check_derived_artifacts as _check_derived_artifacts_impl,
)
from gabion.cli_support.check.check_runtime import run_check as _run_check_impl
from gabion.cli_support.check.execution_plan_payload import (
    ExecutionPlanRequestPayload,
)
from gabion.cli_support.shared import dataflow_runtime_common
from gabion.cli_support.shared.dispatch_runtime import (
    dispatch_command as _dispatch_command_impl,
)
from gabion.cli_support.shared.payload_builder import (
    build_dataflow_payload as _build_dataflow_payload_impl,
)
from gabion.commands import boundary_order, check_contract, command_ids, transport_policy
from gabion.invariants import never
from gabion.json_types import JSONObject
from gabion.lsp_client import CommandRequest, run_command, run_command_direct


Runner = Callable[..., JSONObject]


@dataclass(frozen=True)
class DataflowTransportIngressCarrier:
    stdout_alias: str = dataflow_runtime_common.STDOUT_ALIAS
    stdout_path: str = dataflow_runtime_common.STDOUT_PATH
    default_timeout_ticks: int = dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICKS
    default_timeout_tick_ns: int = dataflow_runtime_common.DEFAULT_CLI_TIMEOUT_TICK_NS

    def cli_timeout_ticks(self) -> tuple[int, int]:
        return dataflow_runtime_common.cli_timeout_ticks(
            default_ticks=self.default_timeout_ticks,
            default_tick_ns=self.default_timeout_tick_ns,
        )

    def resolve_check_report_path(self, report: Path | None, *, root: Path) -> Path:
        return dataflow_runtime_common.resolve_check_report_path(report, root=root)

    def normalize_optional_output_target(self, target: object) -> str | None:
        return dataflow_runtime_common.normalize_optional_output_target(
            target,
            stdout_alias=self.stdout_alias,
            stdout_path=self.stdout_path,
        )

    def build_dataflow_payload_common(
        self,
        *,
        options: check_contract.DataflowPayloadCommonOptions,
    ) -> JSONObject:
        return dataflow_runtime_common.build_dataflow_payload_common(options=options)

    def build_dataflow_payload(self, opts: argparse.Namespace) -> JSONObject:
        return _build_dataflow_payload_impl(
            opts,
            normalize_optional_output_target_fn=self.normalize_optional_output_target,
            build_dataflow_payload_common_fn=self.build_dataflow_payload_common,
        )

    def build_check_execution_plan_request(
        self,
        **kwargs,
    ) -> ExecutionPlanRequestPayload:
        return _build_check_execution_plan_request_impl(
            **kwargs,
            check_derived_artifacts_fn=_check_derived_artifacts_impl,
            execution_plan_request_ctor=ExecutionPlanRequestPayload,
            dataflow_command=command_ids.DATAFLOW_COMMAND,
            check_command=command_ids.CHECK_COMMAND,
        )

    def dispatch_command(
        self,
        *,
        command: str,
        payload: JSONObject,
        root: Path = Path("."),
        runner: Runner = run_command,
        process_factory: Callable[..., subprocess.Popen] | None = None,
        execution_plan_request: ExecutionPlanRequestPayload | None = None,
        notification_callback: Callable[[JSONObject], None] | None = None,
    ) -> JSONObject:
        return _dispatch_command_impl(
            command=command,
            payload=payload,
            root=root,
            runner=runner,
            process_factory=process_factory,
            execution_plan_request=execution_plan_request,
            notification_callback=notification_callback,
            cli_timeout_ticks_fn=self.cli_timeout_ticks,
            normalize_boundary_mapping_once_fn=boundary_order.normalize_boundary_mapping_once,
            apply_boundary_updates_once_fn=boundary_order.apply_boundary_updates_once,
            enforce_boundary_mapping_ordered_fn=boundary_order.enforce_boundary_mapping_ordered,
            command_request_ctor=CommandRequest,
            resolve_command_transport_fn=transport_policy.resolve_command_transport,
            default_lsp_runner=run_command,
            direct_runner=run_command_direct,
            never_fn=never,
        )

    def run_check(self, **kwargs) -> JSONObject:
        runner = kwargs.pop("runner", run_command)
        resolve_check_report_path_fn = kwargs.pop(
            "resolve_check_report_path_fn",
            self.resolve_check_report_path,
        )
        build_check_payload_fn = kwargs.pop(
            "build_check_payload_fn",
            check_contract.build_check_payload,
        )
        build_check_execution_plan_request_fn = kwargs.pop(
            "build_check_execution_plan_request_fn",
            self.build_check_execution_plan_request,
        )
        dispatch_command_fn = kwargs.pop(
            "dispatch_command_fn",
            self.dispatch_command,
        )
        dataflow_command = kwargs.pop("dataflow_command", command_ids.DATAFLOW_COMMAND)
        return _run_check_impl(
            **kwargs,
            runner=runner,
            resolve_check_report_path_fn=resolve_check_report_path_fn,
            build_check_payload_fn=build_check_payload_fn,
            build_check_execution_plan_request_fn=build_check_execution_plan_request_fn,
            dispatch_command_fn=dispatch_command_fn,
            dataflow_command=dataflow_command,
        )


def default_dataflow_transport_ingress() -> DataflowTransportIngressCarrier:
    return DataflowTransportIngressCarrier()


__all__ = [
    "DataflowTransportIngressCarrier",
    "default_dataflow_transport_ingress",
]
