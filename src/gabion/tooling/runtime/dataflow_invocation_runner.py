from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Sequence

from gabion.cli_support.check.execution_plan_payload import (
    ExecutionPlanRequestPayload,
)
from gabion.cli_support.shared.dataflow_transport_ingress import (
    default_dataflow_transport_ingress,
)
from gabion.cli_support.shared.raw_argparse import (
    parse_dataflow_args_or_exit as _parse_dataflow_args_or_exit_impl,
)
from gabion.commands import check_contract, command_ids
from gabion.commands.check_contract import DataflowFilterBundle
from gabion.json_types import JSONObject
from gabion.lsp_client import run_command
from gabion.tooling.runtime.execution_envelope import ExecutionEnvelope

_DATAFLOW_TRANSPORT_INGRESS = default_dataflow_transport_ingress()

def _build_check_execution_plan_request(**kwargs) -> ExecutionPlanRequestPayload:
    return _DATAFLOW_TRANSPORT_INGRESS.build_check_execution_plan_request(**kwargs)


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
        return _DATAFLOW_TRANSPORT_INGRESS.run_check

    def _resolve_dispatch_command(self) -> Callable[..., JSONObject]:
        if callable(self.dispatch_command_fn):
            return self.dispatch_command_fn
        return _DATAFLOW_TRANSPORT_INGRESS.dispatch_command

    def _resolve_parse_dataflow_args(self) -> Callable[[list[str]], argparse.Namespace]:
        if callable(self.parse_dataflow_args_fn):
            return self.parse_dataflow_args_fn
        return _parse_dataflow_args_or_exit_impl

    def _resolve_build_dataflow_payload(
        self,
    ) -> Callable[[argparse.Namespace], JSONObject]:
        if callable(self.build_dataflow_payload_fn):
            return self.build_dataflow_payload_fn
        return _DATAFLOW_TRANSPORT_INGRESS.build_dataflow_payload

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
