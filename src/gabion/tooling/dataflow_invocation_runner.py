# gabion:decision_protocol_module
from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Callable, Sequence

from gabion.analysis.timeout_context import check_deadline
from gabion.commands import check_contract, command_ids
from gabion.commands.check_contract import DataflowFilterBundle
from gabion.json_types import JSONObject
from gabion.tooling.execution_envelope import ExecutionEnvelope


@dataclass(frozen=True)
class DataflowInvocationResult:
    exit_code: int
    analysis_state: str
    payload: JSONObject


@dataclass(frozen=True)
class DataflowInvocationRunner:
    run_check_fn: Callable[..., JSONObject] | None = None
    dispatch_command_fn: Callable[..., JSONObject] | None = None
    cli_module_loader: Callable[[], Any] | None = None

    def _resolve_cli_module(self) -> Any:
        if callable(self.cli_module_loader):
            return self.cli_module_loader()
        return importlib.import_module("gabion.cli")

    def _resolve_run_check(self) -> Callable[..., JSONObject]:
        if callable(self.run_check_fn):
            return self.run_check_fn
        cli_module = self._resolve_cli_module()
        return getattr(cli_module, "run_check")

    def _resolve_dispatch_command(self) -> Callable[..., JSONObject]:
        if callable(self.dispatch_command_fn):
            return self.dispatch_command_fn
        cli_module = self._resolve_cli_module()
        return getattr(cli_module, "dispatch_command")

    # gabion:decision_protocol
    def run_delta_bundle(
        self,
        envelope: ExecutionEnvelope,
    ) -> DataflowInvocationResult:
        checked = envelope.validate()
        if checked.operation != "delta_bundle":
            raise ValueError("execution envelope operation must be 'delta_bundle'")
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
        cli_module = self._resolve_cli_module()
        parse_dataflow_args_or_exit = getattr(cli_module, "parse_dataflow_args_or_exit")
        build_dataflow_payload = getattr(cli_module, "build_dataflow_payload")
        run_command = getattr(cli_module, "run_command")
        dispatch_command = self._resolve_dispatch_command()
        opts = parse_dataflow_args_or_exit(list(raw_args))
        payload = build_dataflow_payload(opts)
        if checked.aspf_state_json is not None:
            payload["aspf_state_json"] = str(checked.aspf_state_json)
            payload["aspf_delta_jsonl"] = str(checked.aspf_delta_jsonl)
            payload["aspf_import_state"] = [str(path) for path in checked.aspf_import_state]
        result = dispatch_command(
            command=command_ids.DATAFLOW_COMMAND,
            payload=payload,
            root=Path(opts.root),
            runner=run_command,
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
