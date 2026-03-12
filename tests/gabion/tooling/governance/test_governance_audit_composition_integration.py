from __future__ import annotations

import argparse
import json
from pathlib import Path

from gabion_governance import governance_audit_impl


# gabion:behavior primary=desired
def test_status_consistency_command_preserves_markdown_json_shape(tmp_path: Path) -> None:
    json_output = tmp_path / "status.json"
    md_output = tmp_path / "status.md"

    args = argparse.Namespace(
        root=str(tmp_path),
        extra_path=None,
        json_output=json_output,
        md_output=md_output,
        fail_on_violations=False,
    )

    rc = governance_audit_impl._status_consistency_command(args)
    assert rc == 0

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert isinstance(summary.get("violation_count"), int)
    assert isinstance(summary.get("warning_count"), int)
    violation_count = int(summary["violation_count"])
    warning_count = int(summary["warning_count"])
    assert isinstance(payload.get("violations"), list)
    assert isinstance(payload.get("warnings"), list)

    markdown = md_output.read_text(encoding="utf-8")
    assert "# SPPF Status Consistency" in markdown
    assert ("## Violations" in markdown) is (violation_count > 0)
    assert ("## Warnings" in markdown) is (warning_count > 0)
    assert ("No issues detected." in markdown) is (violation_count == 0 and warning_count == 0)


# gabion:behavior primary=desired
def test_emit_docflow_canonicality_uses_execution_ops(
    tmp_path: Path,
    monkeypatch,
) -> None:
    json_output = tmp_path / "docflow_canonicality.json"
    md_output = tmp_path / "docflow_canonicality.md"
    entries = [
        {
            "term": "bundle",
            "heading": "Bundle",
            "candidate": False,
            "anchor_present": True,
            "explicit_without_requires": True,
            "requires_without_explicit": False,
            "implicit_docs": [],
            "requires_docs": [],
        }
    ]
    signal_rows = [
        {
            "row_kind": "canonicality_signal",
            "term": "bundle",
            "signal": "explicit_without_requires",
            "doc": "glossary.md#bundle",
        }
    ]
    summary = {
        "total_terms": 1,
        "candidates": 0,
        "ambiguous": 1,
        "no_induced_meaning": 0,
    }

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        governance_audit_impl,
        "_load_docflow_docs",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        governance_audit_impl,
        "_docflow_canonicality_entries",
        lambda **_kwargs: (entries, signal_rows, summary),
    )
    monkeypatch.setattr(
        governance_audit_impl,
        "_docflow_canonicality_execution_ops",
        lambda: ("typed-canonicality-op",),
    )

    def _fake_apply_execution_ops(ops, rows, *, op_registry, runtime_params=None):
        seen["ops"] = ops
        seen["rows"] = rows
        seen["op_registry"] = op_registry
        seen["runtime_params"] = runtime_params
        return [{"term": "bundle", "count": 1}]

    monkeypatch.setattr(
        governance_audit_impl,
        "apply_execution_ops",
        _fake_apply_execution_ops,
    )
    assert not hasattr(governance_audit_impl, "apply_spec")

    governance_audit_impl._emit_docflow_canonicality(
        root=tmp_path,
        extra_paths=None,
        json_output=json_output,
        md_output=md_output,
    )

    assert seen["ops"] == ("typed-canonicality-op",)
    assert seen["rows"] == signal_rows
    op_registry = seen["op_registry"]
    assert isinstance(op_registry, dict)
    assert "canonicality_is_ambiguous" in op_registry
    assert seen["runtime_params"] is None

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["projection_summary"] == [{"count": 1, "term": "bundle"}]
    assert payload["convergence"]["matched"] is True
