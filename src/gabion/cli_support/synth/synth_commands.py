# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable

import typer

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
        root: Path = typer.Option(Path("."), "--root"),
        out_dir: Path = typer.Option(Path("artifacts/synthesis"), "--out-dir"),
        no_timestamp: bool = typer.Option(False, "--no-timestamp"),
        config: Path | None = typer.Option(None, "--config"),
        exclude: list[str] | None = typer.Option(None, "--exclude"),
        ignore_params_csv: str | None = typer.Option(None, "--ignore-params"),
        transparent_decorators_csv: str | None = typer.Option(
            None,
            "--transparent-decorators",
        ),
        allow_external: bool | None = typer.Option(
            None,
            "--allow-external/--no-allow-external",
        ),
        strictness: str | None = typer.Option(None, "--strictness"),
        no_recursive: bool = typer.Option(False, "--no-recursive"),
        max_components: int = typer.Option(10, "--max-components"),
        type_audit_report: bool = typer.Option(
            True,
            "--type-audit-report/--no-type-audit-report",
        ),
        type_audit_max: int = typer.Option(50, "--type-audit-max"),
        synthesis_max_tier: int = typer.Option(2, "--synthesis-max-tier"),
        synthesis_min_bundle_size: int = typer.Option(2, "--synthesis-min-bundle-size"),
        synthesis_allow_singletons: bool = typer.Option(
            False,
            "--synthesis-allow-singletons",
        ),
        synthesis_protocols_kind: str = typer.Option(
            "dataclass",
            "--synthesis-protocols-kind",
        ),
        refactor_plan: bool = typer.Option(True, "--refactor-plan/--no-refactor-plan"),
        fail_on_violations: bool = typer.Option(
            False,
            "--fail-on-violations/--no-fail-on-violations",
        ),
        aspf_trace_json: Path | None = typer.Option(None, "--aspf-trace-json"),
        aspf_import_trace: list[Path] | None = typer.Option(
            None,
            "--aspf-import-trace",
        ),
        aspf_equivalence_against: list[Path] | None = typer.Option(
            None,
            "--aspf-equivalence-against",
        ),
        aspf_opportunities_json: Path | None = typer.Option(
            None,
            "--aspf-opportunities-json",
        ),
        aspf_state_json: Path | None = typer.Option(
            None,
            "--aspf-state-json",
        ),
        aspf_delta_jsonl: Path | None = typer.Option(
            None,
            "--aspf-delta-jsonl",
        ),
        aspf_import_state: list[Path] | None = typer.Option(
            None,
            "--aspf-import-state",
        ),
        aspf_semantic_surface: list[str] | None = typer.Option(
            None,
            "--aspf-semantic-surface",
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
