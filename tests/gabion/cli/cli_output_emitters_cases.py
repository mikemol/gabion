from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Any

from gabion.analysis.dataflow.io import dataflow_output_emitters
from gabion.cli_support.shared.output_emitters import emit_dataflow_result_outputs, write_text_to_target
from gabion.server_core import command_orchestrator_primitives


def _opts(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "lint": None,
        "lint_jsonl": None,
        "lint_sarif": None,
        "type_audit": False,
        "type_audit_max": 10,
        "dot": None,
        "synthesis_plan": None,
        "synthesis_protocols": None,
        "refactor_plan_json": None,
        "fingerprint_synth_json": None,
        "fingerprint_provenance_json": None,
        "fingerprint_deadness_json": None,
        "fingerprint_coherence_json": None,
        "fingerprint_rewrite_plans_json": None,
        "fingerprint_exception_obligations_json": None,
        "fingerprint_handledness_json": None,
        "emit_structure_tree": None,
        "emit_structure_metrics": None,
        "emit_decision_snapshot": None,
        "aspf_trace_json": None,
        "aspf_opportunities_json": None,
        "aspf_state_json": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_emit_dataflow_outputs_uses_canonical_lint_entry_normalization() -> None:
    captured: dict[str, object] = {}

    def _capture_lint(
        _lint_lines: list[str], *, lint: object, lint_jsonl: object, lint_sarif: object, lint_entries: object
    ) -> None:
        captured["lint_entries"] = lint_entries

    emit_dataflow_result_outputs(
        {
            "exit_code": 0,
            "lint_lines": ["pkg/mod.py:1:2: DF001 bad"],
            "lint_entries": "malformed",
        },
        _opts(),
        cli_deadline_scope_factory=lambda: nullcontext(),
        emit_lint_outputs_fn=_capture_lint,
        is_stdout_target_fn=lambda _target: False,
        write_text_to_target_fn=lambda *_args, **_kwargs: None,
        emit_result_json_to_stdout_fn=lambda **_kwargs: None,
        stdout_path="-",
        check_deadline_fn=lambda: None,
        normalize_dataflow_response_fn=command_orchestrator_primitives.normalize_dataflow_response,
        serialize_dataflow_response_fn=command_orchestrator_primitives.serialize_dataflow_response,
    )

    lint_entries = captured["lint_entries"]
    assert isinstance(lint_entries, list)
    assert lint_entries[0]["code"] == "DF001"


def test_emit_dataflow_outputs_respects_canonical_aspf_presence_rules() -> None:
    emitted: list[object] = []

    emit_dataflow_result_outputs(
        {
            "exit_code": 0,
            "aspf_trace": {
                "format_version": 2,
                "trace_id": "aspf-trace:abc123",
                "started_at_utc": "2026-02-25T00:00:00+00:00",
                "controls": {},
                "one_cells": [],
                "two_cell_witnesses": [],
                "cofibration_witnesses": [],
                "surface_representatives": {},
                "imported_trace_count": 0,
            },
            "aspf_opportunities": "malformed",
        },
        _opts(aspf_trace_json="-", aspf_opportunities_json="-", aspf_state_json="-"),
        cli_deadline_scope_factory=lambda: nullcontext(),
        emit_lint_outputs_fn=lambda *_args, **_kwargs: None,
        is_stdout_target_fn=lambda target: target == "-",
        write_text_to_target_fn=lambda *_args, **_kwargs: None,
        emit_result_json_to_stdout_fn=lambda *, payload: emitted.append(payload),
        stdout_path="-",
        check_deadline_fn=lambda: None,
        normalize_dataflow_response_fn=command_orchestrator_primitives.normalize_dataflow_response,
        serialize_dataflow_response_fn=command_orchestrator_primitives.serialize_dataflow_response,
    )

    assert any(isinstance(payload, dict) and payload.get("trace_id") == "aspf-trace:abc123" for payload in emitted)
    assert "malformed" in emitted
    assert all(not (isinstance(payload, dict) and "aspf_state" in payload) for payload in emitted)


def test_emit_dataflow_outputs_uses_canonical_capability_field_normalization() -> None:
    captured: dict[str, object] = {}

    def _normalize(response: dict[str, object]):
        normalized = command_orchestrator_primitives.normalize_dataflow_response(response)
        captured["normalized"] = command_orchestrator_primitives.serialize_dataflow_response(normalized)
        return normalized

    emit_dataflow_result_outputs(
        {
            "exit_code": 0,
            "selected_adapter": 7,
            "supported_analysis_surfaces": "malformed",
            "disabled_surface_reasons": ["malformed"],
        },
        _opts(),
        cli_deadline_scope_factory=lambda: nullcontext(),
        emit_lint_outputs_fn=lambda *_args, **_kwargs: None,
        is_stdout_target_fn=lambda _target: False,
        write_text_to_target_fn=lambda *_args, **_kwargs: None,
        emit_result_json_to_stdout_fn=lambda **_kwargs: None,
        stdout_path="-",
        check_deadline_fn=lambda: None,
        normalize_dataflow_response_fn=_normalize,
        serialize_dataflow_response_fn=command_orchestrator_primitives.serialize_dataflow_response,
    )

    normalized = captured["normalized"]
    assert normalized["selected_adapter"] == "7"
    assert normalized["supported_analysis_surfaces"] == []
    assert normalized["disabled_surface_reasons"] == {}


def test_write_text_to_target_treats_stdout_alias_and_path_equally(capsys) -> None:
    write_text_to_target("-", "alpha", ensure_trailing_newline=True)
    write_text_to_target("/dev/stdout", "beta", ensure_trailing_newline=True)
    captured = capsys.readouterr()
    assert captured.out == "alpha\nbeta\n"


def test_emit_sidecar_outputs_stdout_alias_matches_path(capsys) -> None:
    class _Args:
        fingerprint_synth_json = "-"
        fingerprint_provenance_json = "/dev/stdout"
        lint = False

    class _Analysis:
        fingerprint_synth_registry = {"k": "v"}
        fingerprint_provenance = {"p": 1}
        deadness_witnesses: list[object] = []
        coherence_witnesses: list[object] = []
        rewrite_plans: list[object] = []
        exception_obligations: list[object] = []
        handledness_witnesses: list[object] = []
        lint_lines: list[str] = []

    dataflow_output_emitters.emit_sidecar_outputs(
        args=_Args(),
        analysis=_Analysis(),
        fingerprint_deadness_json=None,
        fingerprint_coherence_json=None,
        fingerprint_rewrite_plans_json=None,
        fingerprint_exception_obligations_json=None,
        fingerprint_handledness_json=None,
    )

    captured = capsys.readouterr()
    assert '{\n  "k": "v"\n}' in captured.out
    assert '{\n  "p": 1\n}' in captured.out
