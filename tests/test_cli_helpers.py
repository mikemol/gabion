from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion import cli


def test_split_csv_helpers() -> None:
    assert cli._split_csv_entries(None) is None
    assert cli._split_csv_entries(["a, b", " ", "c"]) == ["a", "b", "c"]
    assert cli._split_csv_entries([" ", ""]) is None

    assert cli._split_csv(None) is None
    assert cli._split_csv("a, , b") == ["a", "b"]
    assert cli._split_csv(" ,") is None


def test_build_refactor_payload_input_payload_passthrough() -> None:
    payload = {"protocol_name": "Bundle", "bundle": ["a"]}
    assert cli.build_refactor_payload(
        input_payload=payload,
        protocol_name=None,
        bundle=None,
        field=None,
        target_path=None,
        target_functions=None,
        compatibility_shim=False,
        rationale=None,
    ) == payload


def test_build_refactor_payload_requires_fields(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_refactor_payload(
            protocol_name=None,
            bundle=None,
            field=None,
            target_path=None,
            target_functions=None,
            compatibility_shim=False,
            rationale=None,
        )

    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=[" : ", "a:int", "b"],
        target_path=tmp_path / "target.py",
        target_functions=None,
        compatibility_shim=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]
    assert payload["fields"] == [
        {"name": "a", "type_hint": "int"},
        {"name": "b", "type_hint": None},
    ]


def test_run_docflow_audit_missing_script(tmp_path: Path) -> None:
    missing = tmp_path / "missing.py"
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=False,
        script=missing,
    )
    assert exit_code == 2


def test_run_docflow_audit_passes_flags(tmp_path: Path) -> None:
    script = tmp_path / "docflow.py"
    script.write_text("import sys\nsys.exit(0)\n")
    exit_code = cli._run_docflow_audit(
        root=tmp_path,
        fail_on_violations=True,
        script=script,
    )
    assert exit_code == 0


def test_dataflow_audit_skips_type_audit_output() -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    request = cli.DataflowAuditRequest(ctx=DummyCtx(), args=["sample.py"], runner=runner)
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0


def test_dataflow_audit_type_audit_empty_findings() -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0, "type_suggestions": [], "type_ambiguities": []}

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--type-audit", "--type-audit-max", "1"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0


def test_dataflow_audit_emits_structure_tree(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_tree": {"format_version": 1, "root": ".", "files": []},
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-structure-tree", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"format_version\": 1" in captured.out


def test_run_synth_parses_optional_inputs(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {"exit_code": 0}

    result, paths_out, timestamp = cli._run_synth(
        paths=None,
        root=tmp_path,
        out_dir=tmp_path / "out",
        no_timestamp=True,
        config=None,
        exclude=["a, b"],
        ignore_params="x, y",
        transparent_decorators="deco",
        allow_external=None,
        strictness=None,
        no_recursive=False,
        max_components=3,
        type_audit_report=True,
        type_audit_max=5,
        synthesis_max_tier=2,
        synthesis_min_bundle_size=1,
        synthesis_allow_singletons=False,
        synthesis_protocols_kind="dataclass",
        refactor_plan=False,
        fail_on_violations=False,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert timestamp is None
    assert paths_out["output_root"].exists()


def test_run_synthesis_plan_without_input(tmp_path: Path) -> None:
    captured = {}

    def runner(request, *, root=None):
        captured["request"] = request
        captured["root"] = root
        return {"protocols": []}

    output_path = tmp_path / "plan.json"
    cli._run_synthesis_plan(
        input_path=None,
        output_path=output_path,
        runner=runner,
    )
    assert captured["root"] is None
    assert output_path.read_text().strip()
