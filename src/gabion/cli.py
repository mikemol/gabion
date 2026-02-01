from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any
import argparse
import json
import subprocess
import sys

import typer

DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
from gabion.lsp_client import run_command
app = typer.Typer(add_completion=False)


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@app.command()
def check(
    paths: List[Path] = typer.Argument(None),
    report: Optional[Path] = typer.Option(None, "--report"),
    fail_on_violations: bool = typer.Option(True, "--fail-on-violations/--no-fail-on-violations"),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params: Optional[str] = typer.Option(None, "--ignore-params"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
) -> None:
    """Run the dataflow grammar audit with strict defaults."""
    if not paths:
        paths = [Path(".")]
    exclude_dirs: list[str] | None = None
    if exclude is not None:
        exclude_dirs = []
        for entry in exclude:
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list: list[str] | None = None
    if ignore_params is not None:
        ignore_list = [p.strip() for p in ignore_params.split(",") if p.strip()]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    payload = {
        "paths": [str(p) for p in paths],
        "report": str(report) if report is not None else None,
        "fail_on_violations": fail_on_violations,
        "root": str(root),
        "config": str(config) if config is not None else None,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "allow_external": allow_external,
        "strictness": strictness,
    }
    result = run_command(DATAFLOW_COMMAND, [payload])
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command(
    "dataflow-audit",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dataflow_audit(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None),
) -> None:
    """Run the dataflow grammar audit with explicit options."""
    # dataflow-bundle: args, ctx
    argv = list(args or []) + list(ctx.args)
    if not argv:
        argv = []
    parser = dataflow_cli_parser()
    opts = parser.parse_args(argv)
    exclude_dirs: list[str] | None = None
    if opts.exclude is not None:
        exclude_dirs = []
        for entry in opts.exclude:
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list: list[str] | None = None
    if opts.ignore_params is not None:
        ignore_list = [p.strip() for p in opts.ignore_params.split(",") if p.strip()]
    payload: dict[str, Any] = {
        "paths": [str(p) for p in opts.paths],
        "root": str(opts.root),
        "config": str(opts.config) if opts.config is not None else None,
        "report": str(opts.report) if opts.report else None,
        "dot": opts.dot,
        "fail_on_violations": opts.fail_on_violations,
        "no_recursive": opts.no_recursive,
        "max_components": opts.max_components,
        "type_audit": opts.type_audit,
        "type_audit_report": opts.type_audit_report,
        "type_audit_max": opts.type_audit_max,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "allow_external": opts.allow_external,
        "strictness": opts.strictness,
        "synthesis_plan": str(opts.synthesis_plan) if opts.synthesis_plan else None,
        "synthesis_report": opts.synthesis_report,
        "synthesis_max_tier": opts.synthesis_max_tier,
        "synthesis_min_bundle_size": opts.synthesis_min_bundle_size,
        "synthesis_allow_singletons": opts.synthesis_allow_singletons,
        "synthesis_protocols": str(opts.synthesis_protocols)
        if opts.synthesis_protocols
        else None,
    }
    result = run_command(DATAFLOW_COMMAND, [payload])
    if opts.type_audit:
        suggestions = result.get("type_suggestions", [])
        ambiguities = result.get("type_ambiguities", [])
        if suggestions:
            typer.echo("Type tightening candidates:")
            for line in suggestions[: opts.type_audit_max]:
                typer.echo(f"- {line}")
        if ambiguities:
            typer.echo("Type ambiguities (conflicting downstream expectations):")
            for line in ambiguities[: opts.type_audit_max]:
                typer.echo(f"- {line}")
    if opts.dot == "-" and "dot" in result:
        typer.echo(result["dot"])
    if opts.synthesis_plan == "-" and "synthesis_plan" in result:
        typer.echo(json.dumps(result["synthesis_plan"], indent=2, sort_keys=True))
    if opts.synthesis_protocols == "-" and "synthesis_protocols" in result:
        typer.echo(result["synthesis_protocols"])
    raise typer.Exit(code=int(result.get("exit_code", 0)))


def dataflow_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--ignore-params", default=None)
    parser.add_argument(
        "--allow-external",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--strictness", choices=["high", "low"], default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dot", default=None, help="Write DOT graph to file or '-' for stdout.")
    parser.add_argument("--report", default=None, help="Write Markdown report (mermaid) to file.")
    parser.add_argument("--max-components", type=int, default=10, help="Max components in report.")
    parser.add_argument(
        "--type-audit",
        action="store_true",
        help="Emit type-tightening suggestions based on downstream annotations.",
    )
    parser.add_argument(
        "--type-audit-max",
        type=int,
        default=50,
        help="Max type-tightening entries to print.",
    )
    parser.add_argument(
        "--type-audit-report",
        action="store_true",
        help="Include type-flow audit summary in the markdown report.",
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit non-zero if undocumented/undeclared bundle violations are detected.",
    )
    parser.add_argument(
        "--synthesis-plan",
        default=None,
        help="Write synthesis plan JSON to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help="Include synthesis plan summary in the markdown report.",
    )
    parser.add_argument(
        "--synthesis-protocols",
        default=None,
        help="Write protocol/dataclass stubs to file or '-' for stdout.",
    )
    parser.add_argument(
        "--synthesis-max-tier",
        type=int,
        default=2,
        help="Max tier to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-min-bundle-size",
        type=int,
        default=2,
        help="Min bundle size to include in synthesis plan.",
    )
    parser.add_argument(
        "--synthesis-allow-singletons",
        action="store_true",
        help="Allow single-field bundles in synthesis plan.",
    )
    return parser


@app.command("docflow-audit")
def docflow_audit(
    root: Path = typer.Option(Path("."), "--root"),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
) -> None:
    """Run the docflow audit (governance docs only)."""
    repo_root = _find_repo_root()
    script = repo_root / "scripts" / "docflow_audit.py"
    if not script.exists():
        typer.secho(
            "docflow audit script not found; repository layout required",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    args = ["--root", str(root)]
    if fail_on_violations:
        args.append("--fail-on-violations")
    result = subprocess.run([sys.executable, str(script), *args], check=False)
    raise typer.Exit(code=result.returncode)


@app.command("synth")
def synth(
    paths: List[Path] = typer.Argument(None),
    root: Path = typer.Option(Path("."), "--root"),
    out_dir: Path = typer.Option(Path("artifacts/synthesis"), "--out-dir"),
    no_timestamp: bool = typer.Option(False, "--no-timestamp"),
    config: Optional[Path] = typer.Option(None, "--config"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params: Optional[str] = typer.Option(None, "--ignore-params"),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    no_recursive: bool = typer.Option(False, "--no-recursive"),
    max_components: int = typer.Option(10, "--max-components"),
    type_audit_report: bool = typer.Option(
        True, "--type-audit-report/--no-type-audit-report"
    ),
    type_audit_max: int = typer.Option(50, "--type-audit-max"),
    synthesis_max_tier: int = typer.Option(2, "--synthesis-max-tier"),
    synthesis_min_bundle_size: int = typer.Option(2, "--synthesis-min-bundle-size"),
    synthesis_allow_singletons: bool = typer.Option(
        False, "--synthesis-allow-singletons"
    ),
    fail_on_violations: bool = typer.Option(
        False, "--fail-on-violations/--no-fail-on-violations"
    ),
) -> None:
    """Run the dataflow audit and emit synthesis outputs (prototype)."""
    if not paths:
        paths = [Path(".")]
    exclude_dirs: list[str] | None = None
    if exclude is not None:
        exclude_dirs = []
        for entry in exclude:
            exclude_dirs.extend([part.strip() for part in entry.split(",") if part.strip()])
    ignore_list: list[str] | None = None
    if ignore_params is not None:
        ignore_list = [p.strip() for p in ignore_params.split(",") if p.strip()]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")

    output_root = out_dir
    timestamp = None
    if not no_timestamp:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_root = out_dir / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "LATEST.txt").write_text(timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "dataflow_report.md"
    dot_path = output_root / "dataflow_graph.dot"
    plan_path = output_root / "synthesis_plan.json"
    protocol_path = output_root / "protocol_stubs.py"

    payload: dict[str, Any] = {
        "paths": [str(p) for p in paths],
        "root": str(root),
        "config": str(config) if config is not None else None,
        "report": str(report_path),
        "dot": str(dot_path),
        "fail_on_violations": fail_on_violations,
        "no_recursive": no_recursive,
        "max_components": max_components,
        "type_audit_report": type_audit_report,
        "type_audit_max": type_audit_max,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "synthesis_plan": str(plan_path),
        "synthesis_report": True,
        "synthesis_protocols": str(protocol_path),
        "synthesis_max_tier": synthesis_max_tier,
        "synthesis_min_bundle_size": synthesis_min_bundle_size,
        "synthesis_allow_singletons": synthesis_allow_singletons,
    }
    result = run_command(DATAFLOW_COMMAND, [payload])
    if timestamp:
        typer.echo(f"Snapshot: {output_root}")
    typer.echo(f"- {report_path}")
    typer.echo(f"- {dot_path}")
    typer.echo(f"- {plan_path}")
    typer.echo(f"- {protocol_path}")
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command("synthesis-plan")
def synthesis_plan(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing bundles and synthesis settings."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write synthesis plan JSON to this path."
    ),
) -> None:
    """Generate a synthesis plan from a JSON payload (prototype)."""
    payload: dict[str, Any] = {}
    if input_path is not None:
        try:
            payload = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
    result = run_command(SYNTHESIS_COMMAND, [payload])
    output = json.dumps(result, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        output_path.write_text(output)
