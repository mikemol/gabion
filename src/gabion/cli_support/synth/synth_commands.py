from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable, Literal

import typer

from gabion.cli_support.shared.option_schema import SHARED_OPTION_SPECS, typer_option
from gabion.json_types import JSONObject

DataflowFilterBundleCtor = Callable[..., object]
CliDeadlineScopeFactory = Callable[[], AbstractContextManager[object]]
RunSynthFn = Callable[..., tuple[JSONObject, dict[str, Path], Path | None]]
EmitSynthOutputsFn = Callable[..., None]


def register_synth_command(
    *,
    app: typer.Typer,
    dataflow_filter_bundle_ctor: DataflowFilterBundleCtor,
    cli_deadline_scope_factory: CliDeadlineScopeFactory,
    run_synth_fn: RunSynthFn,
    emit_synth_outputs_fn: EmitSynthOutputsFn,
) -> Callable[..., None]:
    @app.command("synth")
    def synth(
        paths: list[Path] = typer.Argument(None),
        root: Path = typer_option(SHARED_OPTION_SPECS["root"]),
        out_dir: Path = typer.Option(Path("artifacts/synthesis"), "--out-dir"),
        no_timestamp: bool = typer.Option(False, "--no-timestamp"),
        config: Path | None = typer_option(SHARED_OPTION_SPECS["config"]),
        exclude: list[str] | None = typer_option(SHARED_OPTION_SPECS["exclude"]),
        ignore_params_csv: str | None = typer_option(SHARED_OPTION_SPECS["ignore_params"]),
        transparent_decorators_csv: str | None = typer_option(
            SHARED_OPTION_SPECS["transparent_decorators"]
        ),
        allow_external: bool | None = typer_option(SHARED_OPTION_SPECS["allow_external"]),
        strictness: Literal["high", "low"] | None = typer_option(SHARED_OPTION_SPECS["strictness"]),
        no_recursive: bool = typer_option(SHARED_OPTION_SPECS["no_recursive"]),
        max_components: int = typer_option(SHARED_OPTION_SPECS["max_components"]),
        type_audit_report: bool = typer.Option(
            True,
            "--type-audit-report/--no-type-audit-report",
        ),
        type_audit_max: int = typer_option(SHARED_OPTION_SPECS["type_audit_max"]),
        synthesis_max_tier: int = typer_option(SHARED_OPTION_SPECS["synthesis_max_tier"]),
        synthesis_min_bundle_size: int = typer_option(
            SHARED_OPTION_SPECS["synthesis_min_bundle_size"]
        ),
        synthesis_allow_singletons: bool = typer_option(
            SHARED_OPTION_SPECS["synthesis_allow_singletons"]
        ),
        synthesis_protocols_kind: Literal["dataclass", "protocol", "contextvar"] = typer_option(
            SHARED_OPTION_SPECS["synthesis_protocols_kind"]
        ),
        refactor_plan: bool = typer.Option(True, "--refactor-plan/--no-refactor-plan"),
        fail_on_violations: bool = typer.Option(
            False,
            "--fail-on-violations/--no-fail-on-violations",
        ),
        aspf_trace_json: Path | None = typer_option(SHARED_OPTION_SPECS["aspf_trace_json"]),
        aspf_import_trace: list[Path] | None = typer_option(SHARED_OPTION_SPECS["aspf_import_trace"]),
        aspf_equivalence_against: list[Path] | None = typer_option(
            SHARED_OPTION_SPECS["aspf_equivalence_against"]
        ),
        aspf_opportunities_json: Path | None = typer_option(
            SHARED_OPTION_SPECS["aspf_opportunities_json"]
        ),
        aspf_state_json: Path | None = typer_option(SHARED_OPTION_SPECS["aspf_state_json"]),
        aspf_delta_jsonl: Path | None = typer_option(SHARED_OPTION_SPECS["aspf_delta_jsonl"]),
        aspf_import_state: list[Path] | None = typer_option(SHARED_OPTION_SPECS["aspf_import_state"]),
        aspf_semantic_surface: list[str] | None = typer_option(
            SHARED_OPTION_SPECS["aspf_semantic_surface"]
        ),
    ) -> None:
        """Run the dataflow audit and emit synthesis outputs (prototype)."""
        with cli_deadline_scope_factory():
            filter_bundle = dataflow_filter_bundle_ctor(
                ignore_params_csv=ignore_params_csv,
                transparent_decorators_csv=transparent_decorators_csv,
            )
            result, paths_out, timestamp = run_synth_fn(
                paths=paths,
                root=root,
                out_dir=out_dir,
                no_timestamp=no_timestamp,
                config=config,
                exclude=exclude,
                filter_bundle=filter_bundle,
                allow_external=allow_external,
                strictness=strictness,
                no_recursive=no_recursive,
                max_components=max_components,
                type_audit_report=type_audit_report,
                type_audit_max=type_audit_max,
                synthesis_max_tier=synthesis_max_tier,
                synthesis_min_bundle_size=synthesis_min_bundle_size,
                synthesis_allow_singletons=synthesis_allow_singletons,
                synthesis_protocols_kind=synthesis_protocols_kind,
                refactor_plan=refactor_plan,
                fail_on_violations=fail_on_violations,
                aspf_trace_json=aspf_trace_json,
                aspf_import_trace=aspf_import_trace,
                aspf_equivalence_against=aspf_equivalence_against,
                aspf_opportunities_json=aspf_opportunities_json,
                aspf_state_json=aspf_state_json,
                aspf_import_state=aspf_import_state,
                aspf_delta_jsonl=aspf_delta_jsonl,
                aspf_semantic_surface=aspf_semantic_surface,
            )
            emit_synth_outputs_fn(
                paths_out=paths_out,
                timestamp=timestamp,
                refactor_plan=refactor_plan,
            )
            raise typer.Exit(code=int(result.get("exit_code", 0)))

    return synth
