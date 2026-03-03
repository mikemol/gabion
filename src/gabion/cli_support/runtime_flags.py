# gabion:boundary_normalization_module
# gabion:decision_protocol_module

from pathlib import Path
from typing import Callable

import typer

ApplyRuntimePolicyFromEnvFn = Callable[[], None]
ApplyCliTimeoutFlagFn = Callable[..., None]
ApplyCliTransportFlagsFn = Callable[..., None]


def register_runtime_flags_callback(
    *,
    app: typer.Typer,
    cli_transport_mode: type,
    apply_runtime_policy_from_env_fn: ApplyRuntimePolicyFromEnvFn,
    apply_cli_timeout_flag_fn: ApplyCliTimeoutFlagFn,
    apply_cli_transport_flags_fn: ApplyCliTransportFlagsFn,
) -> Callable[..., None]:
    @app.callback()
    def configure_runtime_flags(
        timeout: str | None = typer.Option(
            None,
            "--timeout",
            help="Runtime timeout duration (for example: 750ms, 2s, 1m30s).",
        ),
        carrier: cli_transport_mode = typer.Option(
            None,
            "--carrier",
            help="Command transport carrier override.",
        ),
        carrier_override_record: Path | None = typer.Option(
            None,
            "--carrier-override-record",
            help="Path to override lifecycle record for direct carrier on governed commands.",
        ),
        removed_lsp_timeout_ticks: int | None = typer.Option(
            None,
            "--lsp-timeout-ticks",
            hidden=True,
        ),
        removed_lsp_timeout_tick_ns: int | None = typer.Option(
            None,
            "--lsp-timeout-tick-ns",
            hidden=True,
        ),
        removed_lsp_timeout_ms: int | None = typer.Option(
            None,
            "--lsp-timeout-ms",
            hidden=True,
        ),
        removed_lsp_timeout_seconds: float | None = typer.Option(
            None,
            "--lsp-timeout-seconds",
            hidden=True,
        ),
        removed_transport: str | None = typer.Option(
            None,
            "--transport",
            hidden=True,
        ),
        removed_direct_run_override_evidence: str | None = typer.Option(
            None,
            "--direct-run-override-evidence",
            hidden=True,
        ),
        removed_override_record_json: str | None = typer.Option(
            None,
            "--override-record-json",
            hidden=True,
        ),
    ) -> None:
        if (
            removed_lsp_timeout_ticks is not None
            or removed_lsp_timeout_tick_ns is not None
            or removed_lsp_timeout_ms is not None
            or removed_lsp_timeout_seconds is not None
        ):
            raise typer.BadParameter(
                "Removed timeout flags (--lsp-timeout-*). Use --timeout <duration>."
            )
        if (
            removed_transport is not None
            or removed_direct_run_override_evidence is not None
            or removed_override_record_json is not None
        ):
            raise typer.BadParameter(
                "Removed transport flags (--transport/--direct-run-override-evidence/--override-record-json). "
                "Use --carrier and --carrier-override-record."
            )
        apply_runtime_policy_from_env_fn()
        apply_cli_timeout_flag_fn(timeout=timeout)
        if carrier is not None or carrier_override_record is not None:
            carrier_text = None if carrier is None else str(carrier.value)
            carrier_override_record_text = (
                None
                if carrier_override_record is None
                else str(carrier_override_record)
            )
            apply_cli_transport_flags_fn(
                carrier=carrier_text,
                override_record_path=carrier_override_record_text,
            )

    return configure_runtime_flags
