from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Callable
import argparse
import json
import subprocess
import sys

import typer

DATAFLOW_COMMAND = "gabion.dataflowAudit"
SYNTHESIS_COMMAND = "gabion.synthesisPlan"
REFACTOR_COMMAND = "gabion.refactorProtocol"
from gabion.lsp_client import CommandRequest, run_command
app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class DataflowAuditRequest:
    ctx: typer.Context
    args: List[str] | None = None


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_csv_entries(entries: Optional[List[str]]) -> list[str] | None:
    if entries is None:
        return None
    merged: list[str] = []
    for entry in entries:
        merged.extend([part.strip() for part in entry.split(",") if part.strip()])
    return merged or None


def _split_csv(value: Optional[str]) -> list[str] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def build_check_payload(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    ignore_params: Optional[str],
    transparent_decorators: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
) -> dict[str, Any]:
    if not paths:
        paths = [Path(".")]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    exclude_dirs = _split_csv_entries(exclude)
    ignore_list = _split_csv(ignore_params)
    transparent_list = _split_csv(transparent_decorators)
    baseline_write_value: bool | None = baseline_write if baseline is not None else None
    payload = {
        "paths": [str(p) for p in paths],
        "report": str(report) if report is not None else None,
        "fail_on_violations": fail_on_violations,
        "fail_on_type_ambiguities": fail_on_type_ambiguities,
        "root": str(root),
        "config": str(config) if config is not None else None,
        "baseline": str(baseline) if baseline is not None else None,
        "baseline_write": baseline_write_value,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "type_audit": True if fail_on_type_ambiguities else None,
    }
    return payload


def parse_dataflow_args(argv: list[str]) -> argparse.Namespace:
    parser = dataflow_cli_parser()
    return parser.parse_args(argv)


def build_dataflow_payload(opts: argparse.Namespace) -> dict[str, Any]:
    exclude_dirs = _split_csv_entries(opts.exclude)
    ignore_list = _split_csv(opts.ignore_params)
    transparent_list = _split_csv(opts.transparent_decorators)
    payload: dict[str, Any] = {
        "paths": [str(p) for p in opts.paths],
        "root": str(opts.root),
        "config": str(opts.config) if opts.config is not None else None,
        "report": str(opts.report) if opts.report else None,
        "dot": opts.dot,
        "fail_on_violations": opts.fail_on_violations,
        "fail_on_type_ambiguities": opts.fail_on_type_ambiguities,
        "baseline": str(opts.baseline) if opts.baseline else None,
        "baseline_write": opts.baseline_write if opts.baseline else None,
        "no_recursive": opts.no_recursive,
        "max_components": opts.max_components,
        "type_audit": opts.type_audit,
        "type_audit_report": opts.type_audit_report,
        "type_audit_max": opts.type_audit_max,
        "exclude": exclude_dirs,
        "ignore_params": ignore_list,
        "transparent_decorators": transparent_list,
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
        "synthesis_protocols_kind": opts.synthesis_protocols_kind,
        "refactor_plan": opts.refactor_plan,
        "refactor_plan_json": str(opts.refactor_plan_json)
        if opts.refactor_plan_json
        else None,
        "synthesis_merge_overlap": opts.synthesis_merge_overlap,
    }
    return payload


def build_refactor_payload(
    *,
    input_payload: Optional[dict[str, Any]] = None,
    protocol_name: Optional[str],
    bundle: Optional[List[str]],
    field: Optional[List[str]],
    target_path: Optional[Path],
    target_functions: Optional[List[str]],
    compatibility_shim: bool,
    rationale: Optional[str],
) -> dict[str, Any]:
    if input_payload is not None:
        return input_payload
    if protocol_name is None or target_path is None:
        raise typer.BadParameter(
            "Provide --protocol-name and --target-path or use --input."
        )
    field_specs: list[dict[str, str | None]] = []
    for spec in field or []:
        name, _, hint = spec.partition(":")
        name = name.strip()
        if not name:
            continue
        type_hint = hint.strip() or None
        field_specs.append({"name": name, "type_hint": type_hint})
    if not bundle and field_specs:
        bundle = [spec["name"] for spec in field_specs]
    return {
        "protocol_name": protocol_name,
        "bundle": bundle or [],
        "fields": field_specs,
        "target_path": str(target_path),
        "target_functions": target_functions or [],
        "compatibility_shim": compatibility_shim,
        "rationale": rationale,
    }


def dispatch_command(
    *,
    command: str,
    payload: dict[str, Any],
    root: Path | None = None,
    runner: Callable[..., dict[str, Any]] = run_command,
) -> dict[str, Any]:
    request = CommandRequest(command, [payload])
    return runner(request, root=root)


def run_check(
    *,
    paths: Optional[List[Path]],
    report: Optional[Path],
    fail_on_violations: bool,
    root: Path,
    config: Optional[Path],
    baseline: Optional[Path],
    baseline_write: bool,
    exclude: Optional[List[str]],
    ignore_params: Optional[str],
    transparent_decorators: Optional[str],
    allow_external: Optional[bool],
    strictness: Optional[str],
    fail_on_type_ambiguities: bool,
    runner: Callable[..., dict[str, Any]] = run_command,
) -> dict[str, Any]:
    payload = build_check_payload(
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write if baseline is not None else False,
        exclude=exclude,
        ignore_params=ignore_params,
        transparent_decorators=transparent_decorators,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
    )
    return dispatch_command(command=DATAFLOW_COMMAND, payload=payload, root=root, runner=runner)


@app.command()
def check(
    paths: List[Path] = typer.Argument(None),
    report: Optional[Path] = typer.Option(None, "--report"),
    fail_on_violations: bool = typer.Option(True, "--fail-on-violations/--no-fail-on-violations"),
    root: Path = typer.Option(Path("."), "--root"),
    config: Optional[Path] = typer.Option(None, "--config"),
    baseline: Optional[Path] = typer.Option(
        None, "--baseline", help="Baseline file of allowed violations."
    ),
    baseline_write: bool = typer.Option(
        False, "--baseline-write", help="Write current violations to baseline."
    ),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude"),
    ignore_params: Optional[str] = typer.Option(None, "--ignore-params"),
    transparent_decorators: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
    allow_external: Optional[bool] = typer.Option(
        None, "--allow-external/--no-allow-external"
    ),
    strictness: Optional[str] = typer.Option(None, "--strictness"),
    fail_on_type_ambiguities: bool = typer.Option(
        True, "--fail-on-type-ambiguities/--no-fail-on-type-ambiguities"
    ),
) -> None:
    """Run the dataflow grammar audit with strict defaults."""
    result = run_check(
        paths=paths,
        report=report,
        fail_on_violations=fail_on_violations,
        root=root,
        config=config,
        baseline=baseline,
        baseline_write=baseline_write,
        exclude=exclude,
        ignore_params=ignore_params,
        transparent_decorators=transparent_decorators,
        allow_external=allow_external,
        strictness=strictness,
        fail_on_type_ambiguities=fail_on_type_ambiguities,
    )
    raise typer.Exit(code=int(result.get("exit_code", 0)))


def _dataflow_audit(
    request: "DataflowAuditRequest",
) -> None:
    """Run the dataflow grammar audit with explicit options."""
    argv = list(request.args or []) + list(request.ctx.args)
    if not argv:
        argv = []
    opts = parse_dataflow_args(argv)
    payload = build_dataflow_payload(opts)
    result = dispatch_command(
        command=DATAFLOW_COMMAND,
        payload=payload,
        root=Path(opts.root),
    )
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
    if opts.refactor_plan_json == "-" and "refactor_plan" in result:
        typer.echo(json.dumps(result["refactor_plan"], indent=2, sort_keys=True))
    raise typer.Exit(code=int(result.get("exit_code", 0)))


@app.command(
    "dataflow-audit",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def dataflow_audit(
    ctx: typer.Context,
    args: List[str] = typer.Argument(None),
) -> None:
    request = DataflowAuditRequest(ctx=ctx, args=args)
    _dataflow_audit(request)


def dataflow_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline", default=None, help="Baseline file for violations.")
    parser.add_argument(
        "--baseline-write",
        action="store_true",
        help="Write current violations to baseline file.",
    )
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--ignore-params", default=None)
    parser.add_argument(
        "--transparent-decorators",
        default=None,
        help="Comma-separated decorator names treated as transparent.",
    )
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
        "--fail-on-type-ambiguities",
        action="store_true",
        help="Exit non-zero if type ambiguities are detected.",
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
        "--synthesis-protocols-kind",
        choices=["dataclass", "protocol"],
        default="dataclass",
        help="Emit dataclass or typing.Protocol stubs (default: dataclass).",
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
    parser.add_argument(
        "--synthesis-merge-overlap",
        type=float,
        default=None,
        help="Jaccard overlap threshold for merging bundles (0.0-1.0).",
    )
    parser.add_argument(
        "--refactor-plan",
        action="store_true",
        help="Include refactoring plan summary in the markdown report.",
    )
    parser.add_argument(
        "--refactor-plan-json",
        default=None,
        help="Write refactoring plan JSON to file or '-' for stdout.",
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
    transparent_decorators: Optional[str] = typer.Option(
        None, "--transparent-decorators"
    ),
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
    synthesis_protocols_kind: str = typer.Option(
        "dataclass", "--synthesis-protocols-kind"
    ),
    refactor_plan: bool = typer.Option(True, "--refactor-plan/--no-refactor-plan"),
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
    transparent_list: list[str] | None = None
    if transparent_decorators is not None:
        transparent_list = [
            p.strip() for p in transparent_decorators.split(",") if p.strip()
        ]
    if strictness is not None and strictness not in {"high", "low"}:
        raise typer.BadParameter("strictness must be 'high' or 'low'")
    if synthesis_protocols_kind not in {"dataclass", "protocol"}:
        raise typer.BadParameter(
            "synthesis-protocols-kind must be 'dataclass' or 'protocol'"
        )

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
    refactor_plan_path = output_root / "refactor_plan.json"

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
        "transparent_decorators": transparent_list,
        "allow_external": allow_external,
        "strictness": strictness,
        "synthesis_plan": str(plan_path),
        "synthesis_report": True,
        "synthesis_protocols": str(protocol_path),
        "synthesis_protocols_kind": synthesis_protocols_kind,
        "synthesis_max_tier": synthesis_max_tier,
        "synthesis_min_bundle_size": synthesis_min_bundle_size,
        "synthesis_allow_singletons": synthesis_allow_singletons,
        "refactor_plan": refactor_plan,
        "refactor_plan_json": str(refactor_plan_path) if refactor_plan else None,
    }
    result = dispatch_command(
        command=DATAFLOW_COMMAND,
        payload=payload,
        root=root,
    )
    if timestamp:
        typer.echo(f"Snapshot: {output_root}")
    typer.echo(f"- {report_path}")
    typer.echo(f"- {dot_path}")
    typer.echo(f"- {plan_path}")
    typer.echo(f"- {protocol_path}")
    if refactor_plan:
        typer.echo(f"- {refactor_plan_path}")
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
    result = dispatch_command(
        command=SYNTHESIS_COMMAND,
        payload=payload,
        root=None,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        output_path.write_text(output)


@app.command("refactor-protocol")
def refactor_protocol(
    input_path: Optional[Path] = typer.Option(
        None, "--input", help="JSON payload describing the refactor request."
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output", help="Write refactor response JSON to this path."
    ),
    protocol_name: Optional[str] = typer.Option(None, "--protocol-name"),
    bundle: Optional[List[str]] = typer.Option(None, "--bundle"),
    field: Optional[List[str]] = typer.Option(
        None,
        "--field",
        help="Field spec in 'name:type' form (repeatable).",
    ),
    target_path: Optional[Path] = typer.Option(None, "--target-path"),
    target_functions: Optional[List[str]] = typer.Option(None, "--target-function"),
    compatibility_shim: bool = typer.Option(
        False, "--compat-shim/--no-compat-shim"
    ),
    rationale: Optional[str] = typer.Option(None, "--rationale"),
) -> None:
    """Generate protocol refactor edits from a JSON payload (prototype)."""
    input_payload: dict[str, Any] | None = None
    if input_path is not None:
        try:
            input_payload = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
    payload = build_refactor_payload(
        input_payload=input_payload,
        protocol_name=protocol_name,
        bundle=bundle,
        field=field,
        target_path=target_path,
        target_functions=target_functions,
        compatibility_shim=compatibility_shim,
        rationale=rationale,
    )
    result = dispatch_command(
        command=REFACTOR_COMMAND,
        payload=payload,
        root=None,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if output_path is None:
        typer.echo(output)
    else:
        output_path.write_text(output)
