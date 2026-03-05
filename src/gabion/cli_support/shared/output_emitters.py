# gabion:boundary_normalization_module
# gabion:decision_protocol_module
from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Callable

import typer

from gabion.json_types import JSONObject
from gabion.schema import DataflowResponseEnvelopeDTO
from gabion.runtime import path_policy

DeadlineScopeFactory = Callable[[], AbstractContextManager[object]]
EmitLintOutputsFn = Callable[..., None]
IsStdoutTargetFn = Callable[[object], bool]
WriteTextToTargetFn = Callable[..., None]
EmitResultJsonToStdoutFn = Callable[..., None]
CheckDeadlineFn = Callable[[], None]
SortOnceFn = Callable[..., list[str]]
NormalizeDataflowResponseFn = Callable[[JSONObject], DataflowResponseEnvelopeDTO]
SerializeDataflowResponseFn = Callable[[DataflowResponseEnvelopeDTO], dict[str, object]]


def normalize_output_target(
    target: str | Path,
    *,
    stdout_alias: str = path_policy.STDOUT_ALIAS,
    stdout_path: str = path_policy.STDOUT_PATH,
) -> str:
    return path_policy.normalize_output_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def is_stdout_target(
    target: object,
    *,
    stdout_alias: str = path_policy.STDOUT_ALIAS,
    stdout_path: str = path_policy.STDOUT_PATH,
) -> bool:
    return path_policy.is_stdout_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def write_text_to_target(
    target: str | Path,
    payload: str,
    *,
    ensure_trailing_newline: bool = False,
    encoding: str = "utf-8",
    stdout_alias: str = path_policy.STDOUT_ALIAS,
    stdout_path: str = path_policy.STDOUT_PATH,
) -> None:
    normalized_target = normalize_output_target(
        target,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )
    text = payload
    if ensure_trailing_newline and not text.endswith("\n"):
        text = f"{text}\n"
    if normalized_target == stdout_path:
        print(text, end="")
        return
    Path(normalized_target).write_text(text, encoding=encoding)


def write_json_to_target(
    target: str | Path,
    payload: object,
    *,
    ensure_trailing_newline: bool = True,
    stdout_alias: str = path_policy.STDOUT_ALIAS,
    stdout_path: str = path_policy.STDOUT_PATH,
) -> None:
    write_text_to_target(
        target,
        json.dumps(payload, indent=2, sort_keys=False),
        ensure_trailing_newline=ensure_trailing_newline,
        stdout_alias=stdout_alias,
        stdout_path=stdout_path,
    )


def emit_dataflow_result_outputs(
    result: JSONObject,
    opts: argparse.Namespace,
    *,
    cli_deadline_scope_factory: DeadlineScopeFactory,
    emit_lint_outputs_fn: EmitLintOutputsFn,
    is_stdout_target_fn: IsStdoutTargetFn,
    write_text_to_target_fn: WriteTextToTargetFn,
    emit_result_json_to_stdout_fn: EmitResultJsonToStdoutFn,
    stdout_path: str,
    check_deadline_fn: CheckDeadlineFn,
    normalize_dataflow_response_fn: NormalizeDataflowResponseFn,
    serialize_dataflow_response_fn: SerializeDataflowResponseFn,
) -> None:
    with cli_deadline_scope_factory():
        normalized_envelope = normalize_dataflow_response_fn(result)
        canonical = normalized_envelope.canonical
        normalized_result = serialize_dataflow_response_fn(normalized_envelope)
        lint_lines = list(canonical.lint_lines)
        lint_entries_raw = normalized_result.get("lint_entries")
        lint_entries = lint_entries_raw if isinstance(lint_entries_raw, list) else None
        emit_lint_outputs_fn(
            lint_lines,
            lint=opts.lint,
            lint_jsonl=opts.lint_jsonl,
            lint_sarif=opts.lint_sarif,
            lint_entries=lint_entries,
        )
        if opts.type_audit:
            suggestions = normalized_result.get("type_suggestions", [])
            ambiguities = normalized_result.get("type_ambiguities", [])
            if suggestions:
                typer.echo("Type tightening candidates:")
                for line in suggestions[: opts.type_audit_max]:
                    check_deadline_fn()
                    typer.echo(f"- {line}")
            if ambiguities:
                typer.echo("Type ambiguities (conflicting downstream expectations):")
                for line in ambiguities[: opts.type_audit_max]:
                    check_deadline_fn()
                    typer.echo(f"- {line}")
        if is_stdout_target_fn(opts.dot) and "dot" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                str(normalized_result["dot"]),
                ensure_trailing_newline=True,
            )
        if is_stdout_target_fn(opts.synthesis_plan) and "synthesis_plan" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["synthesis_plan"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if is_stdout_target_fn(opts.synthesis_protocols) and "synthesis_protocols" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                str(normalized_result["synthesis_protocols"]),
                ensure_trailing_newline=True,
            )
        if is_stdout_target_fn(opts.refactor_plan_json) and "refactor_plan" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["refactor_plan"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            is_stdout_target_fn(opts.fingerprint_synth_json)
            and "fingerprint_synth_registry" in normalized_result
        ):
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_synth_registry"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            is_stdout_target_fn(opts.fingerprint_provenance_json)
            and "fingerprint_provenance" in normalized_result
        ):
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_provenance"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if is_stdout_target_fn(opts.fingerprint_deadness_json) and "fingerprint_deadness" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_deadness"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if is_stdout_target_fn(opts.fingerprint_coherence_json) and "fingerprint_coherence" in normalized_result:
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_coherence"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            is_stdout_target_fn(opts.fingerprint_rewrite_plans_json)
            and "fingerprint_rewrite_plans" in normalized_result
        ):
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_rewrite_plans"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        if (
            is_stdout_target_fn(opts.fingerprint_exception_obligations_json)
            and "fingerprint_exception_obligations" in normalized_result
        ):
            write_text_to_target_fn(
                stdout_path,
                json.dumps(
                    normalized_result["fingerprint_exception_obligations"],
                    indent=2,
                    sort_keys=False,
                ),
                ensure_trailing_newline=True,
            )
        if (
            is_stdout_target_fn(opts.fingerprint_handledness_json)
            and "fingerprint_handledness" in normalized_result
        ):
            write_text_to_target_fn(
                stdout_path,
                json.dumps(normalized_result["fingerprint_handledness"], indent=2, sort_keys=False),
                ensure_trailing_newline=True,
            )
        stdout_json_targets = (
            (opts.emit_structure_tree, "structure_tree"),
            (opts.emit_structure_metrics, "structure_metrics"),
            (opts.emit_decision_snapshot, "decision_snapshot"),
            (opts.aspf_trace_json, "aspf_trace"),
            (opts.aspf_trace_json, "aspf_equivalence"),
            (opts.aspf_opportunities_json, "aspf_opportunities"),
            (opts.aspf_state_json, "aspf_state"),
        )
        for output_target, result_key in stdout_json_targets:
            if result_key in normalized_result and is_stdout_target_fn(output_target):
                emit_result_json_to_stdout_fn(payload=normalized_result[result_key])


def write_lint_sarif(
    target: str,
    entries: list[dict[str, object]],
    *,
    check_deadline_fn: CheckDeadlineFn,
    sort_once_fn: SortOnceFn,
    write_text_to_target_fn: WriteTextToTargetFn,
) -> None:
    check_deadline_fn()
    rules: dict[str, dict[str, object]] = {}
    rule_counts: dict[str, int] = {}
    results: list[dict[str, object]] = []
    for entry in entries:
        check_deadline_fn()
        code = str(entry.get("code") or "GABION")
        message = str(entry.get("message") or "").strip()
        path = str(entry.get("path") or "")
        line = int(entry.get("line") or 1)
        col = int(entry.get("col") or 1)
        rule_counts[code] = int(rule_counts.get(code, 0)) + 1
        rules[code] = {
            "id": code,
            "name": code,
            "shortDescription": {"text": code},
        }
        results.append(
            {
                "ruleId": code,
                "level": "warning",
                "message": {"text": message or code},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": path},
                            "region": {
                                "startLine": line,
                                "startColumn": col,
                            },
                        }
                    }
                ],
            }
        )
    duplicate_codes = sort_once_fn(
        (code for code, count in rule_counts.items() if int(count) > 1),
        source="gabion.cli._emit_lint_sarif.duplicate_codes",
    )
    if duplicate_codes:
        joined = ", ".join(duplicate_codes)
        raise ValueError(f"duplicate SARIF rule code(s): {joined}")
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {"driver": {"name": "gabion", "rules": list(rules.values())}},
                "results": results,
            }
        ],
    }
    payload = json.dumps(sarif, indent=2, sort_keys=False)
    write_text_to_target_fn(target, payload, ensure_trailing_newline=True)
