# gabion:decision_protocol_module
from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Protocol

from gabion.json_types import JSONObject, JSONValue

Runner = Callable[..., JSONObject]


class _ExecutionPlanRequest(Protocol):
    def to_payload(self) -> JSONObject: ...


class _ResolvedTransport(Protocol):
    runner: Runner


def dispatch_command(
    *,
    command: str,
    payload: JSONObject,
    root: Path = Path("."),
    runner: Runner,
    process_factory: Callable[..., subprocess.Popen] | None = None,
    execution_plan_request: _ExecutionPlanRequest | None = None,
    notification_callback: Callable[[JSONObject], None] | None = None,
    cli_timeout_ticks_fn: Callable[[], tuple[int, int]],
    normalize_boundary_mapping_once_fn: Callable[..., JSONObject],
    apply_boundary_updates_once_fn: Callable[..., JSONObject],
    enforce_boundary_mapping_ordered_fn: Callable[..., JSONObject],
    command_request_ctor: Callable[[str, list[JSONObject]], object],
    resolve_command_transport_fn: Callable[..., _ResolvedTransport],
    default_lsp_runner: Runner,
    direct_runner: Runner,
    never_fn: Callable[..., object],
) -> JSONObject:
    def _ordered_result(
        value: Mapping[str, JSONValue],
    ) -> JSONObject:
        return normalize_boundary_mapping_once_fn(
            value,
            source=f"cli.dispatch_command.{command}.result",
        )

    ticks, tick_ns = cli_timeout_ticks_fn()
    payload = normalize_boundary_mapping_once_fn(
        payload,
        source=f"cli.dispatch_command.{command}.payload_in",
    )
    if (
        "analysis_timeout_ticks" not in payload
        and "analysis_timeout_ms" not in payload
        and "analysis_timeout_seconds" not in payload
    ):
        payload = apply_boundary_updates_once_fn(
            payload,
            {
                "analysis_timeout_ticks": int(ticks),
                "analysis_timeout_tick_ns": int(tick_ns),
            },
            source=f"cli.dispatch_command.{command}.payload_timeout_defaults",
        )
    if execution_plan_request is not None:
        execution_plan_payload = execution_plan_request.to_payload()
        execution_plan_inputs = execution_plan_payload.get("inputs")
        if isinstance(execution_plan_inputs, Mapping):
            merged_inputs = apply_boundary_updates_once_fn(
                execution_plan_inputs,
                payload,
                source=f"cli.dispatch_command.{command}.execution_plan_inputs",
            )
            execution_plan_payload["inputs"] = merged_inputs
        deadline_metadata = execution_plan_payload.get("policy_metadata")
        if isinstance(deadline_metadata, Mapping):
            policy_metadata = dict(deadline_metadata)
            deadline = policy_metadata.get("deadline")
            deadline_payload = dict(deadline) if isinstance(deadline, Mapping) else {}
            deadline_payload = apply_boundary_updates_once_fn(
                deadline_payload,
                {
                    "analysis_timeout_ticks": int(
                        payload.get("analysis_timeout_ticks") or 0
                    ),
                    "analysis_timeout_tick_ns": int(
                        payload.get("analysis_timeout_tick_ns") or 0
                    ),
                },
                source=f"cli.dispatch_command.{command}.execution_plan_deadline",
            )
            policy_metadata["deadline"] = deadline_payload
            execution_plan_payload["policy_metadata"] = policy_metadata
        execution_plan_payload = normalize_boundary_mapping_once_fn(
            execution_plan_payload,
            source=f"cli.dispatch_command.{command}.execution_plan_payload",
        )
        payload = apply_boundary_updates_once_fn(
            payload,
            {"execution_plan_request": execution_plan_payload},
            source=f"cli.dispatch_command.{command}.payload_execution_plan",
        )
    payload = enforce_boundary_mapping_ordered_fn(
        payload,
        source=f"cli.dispatch_command.{command}.payload_out",
    )
    request = command_request_ctor(command, [payload])
    transport = resolve_command_transport_fn(
        command=command,
        runner=runner,
        default_lsp_runner=default_lsp_runner,
        direct_runner=direct_runner,
    )
    resolved = transport.runner
    if resolved is default_lsp_runner:
        factory = process_factory or subprocess.Popen
        raw = resolved(
            request,
            root=root,
            timeout_ticks=ticks,
            timeout_tick_ns=tick_ns,
            process_factory=factory,
            notification_callback=notification_callback,
        )
    elif resolved is direct_runner:
        raw = resolved(
            request,
            root=root,
            notification_callback=notification_callback,
        )
    else:
        if notification_callback is not None:
            try:
                params = inspect.signature(resolved).parameters
            except (TypeError, ValueError):
                params = {}
            if "notification_callback" in params or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            ):
                raw = resolved(
                    request,
                    root=root,
                    notification_callback=notification_callback,
                )
            else:
                raw = resolved(request, root=root)
        else:
            raw = resolved(request, root=root)
    if not isinstance(raw, Mapping):
        never_fn(
            "command returned non-mapping payload",
            command=command,
            result_type=type(raw).__name__,
        )
    return _ordered_result(raw)
