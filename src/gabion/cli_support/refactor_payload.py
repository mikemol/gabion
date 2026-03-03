# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer

from gabion.json_types import JSONObject

CheckDeadlineFn = Callable[[], None]


def build_refactor_payload(
    *,
    input_payload: JSONObject | None = None,
    protocol_name: str | None,
    bundle: list[str] | None,
    field: list[str] | None,
    target_path: Path | None,
    target_functions: list[str] | None,
    compatibility_shim: bool,
    compatibility_shim_warnings: bool,
    compatibility_shim_overloads: bool,
    ambient_rewrite: bool,
    rationale: str | None,
    check_deadline_fn: CheckDeadlineFn,
) -> JSONObject:
    check_deadline_fn()
    if input_payload is not None:
        return input_payload
    if protocol_name is None or target_path is None:
        raise typer.BadParameter(
            "Provide --protocol-name and --target-path or use --input."
        )
    field_specs: list[dict[str, str | None]] = []
    for spec in field or []:
        check_deadline_fn()
        name, _, hint = spec.partition(":")
        name = name.strip()
        if not name:
            continue
        type_hint = hint.strip() or None
        field_specs.append({"name": name, "type_hint": type_hint})
    if not bundle and field_specs:
        bundle = [spec["name"] for spec in field_specs]
    compatibility_shim_payload: bool | dict[str, bool]
    if compatibility_shim:
        compatibility_shim_payload = {
            "enabled": True,
            "emit_deprecation_warning": compatibility_shim_warnings,
            "emit_overload_stubs": compatibility_shim_overloads,
        }
    else:
        compatibility_shim_payload = False
    return {
        "protocol_name": protocol_name,
        "bundle": bundle or [],
        "fields": field_specs,
        "target_path": str(target_path),
        "target_functions": target_functions or [],
        "compatibility_shim": compatibility_shim_payload,
        "ambient_rewrite": ambient_rewrite,
        "rationale": rationale,
    }
