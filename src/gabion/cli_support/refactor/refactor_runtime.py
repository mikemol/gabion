from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import typer

from gabion.json_types import JSONObject
from gabion.lsp_client import run_command
from gabion.runtime_shape_dispatch import json_mapping_optional

Runner = Callable[..., JSONObject]


def run_refactor_protocol(
    *,
    input_path: Path | None,
    output_path: Path | None,
    rewrite_kind: str = "protocol_extract",
    protocol_name: str | None,
    bundle: list[str] | None,
    field: list[str] | None,
    target_path: Path | None,
    target_functions: list[str] | None,
    target_loop_lines: list[int] | None = None,
    compatibility_shim: bool,
    compatibility_shim_warnings: bool,
    compatibility_shim_overloads: bool,
    ambient_rewrite: bool,
    rationale: str | None,
    runner: Runner = run_command,
    build_refactor_payload_fn: Callable[..., JSONObject],
    dispatch_command_fn: Callable[..., JSONObject],
    refactor_command: str,
    response_model_validate_fn: Callable[[JSONObject], object],
    write_text_to_target_fn: Callable[..., None],
) -> None:
    input_payload: JSONObject | None = None
    if input_path is not None:
        try:
            loaded = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"Invalid JSON payload: {exc}") from exc
        normalized_loaded = json_mapping_optional(loaded)
        if normalized_loaded is None:
            raise typer.BadParameter("Refactor payload must be a JSON object.")
        input_payload = {str(key): normalized_loaded[key] for key in normalized_loaded}
    payload = build_refactor_payload_fn(
        input_payload=input_payload,
        rewrite_kind=rewrite_kind,
        protocol_name=protocol_name,
        bundle=bundle,
        field=field,
        target_path=target_path,
        target_functions=target_functions,
        target_loop_lines=target_loop_lines,
        compatibility_shim=compatibility_shim,
        compatibility_shim_warnings=compatibility_shim_warnings,
        compatibility_shim_overloads=compatibility_shim_overloads,
        ambient_rewrite=ambient_rewrite,
        rationale=rationale,
    )
    result = dispatch_command_fn(
        command=refactor_command,
        payload=payload,
        root=None,
        runner=runner,
    )
    normalized = response_model_validate_fn(result)
    if hasattr(normalized, "model_dump"):
        normalized = normalized.model_dump()
    output = json.dumps(normalized, indent=2, sort_keys=False)
    if output_path is None:
        typer.echo(output)
    else:
        write_text_to_target_fn(output_path, output)
