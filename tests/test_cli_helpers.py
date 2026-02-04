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


def test_dataflow_audit_emits_structure_metrics(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "structure_metrics": {
                "files": 0,
                "functions": 0,
                "bundles": 0,
                "mean_bundle_size": 0.0,
                "max_bundle_size": 0,
                "bundle_size_histogram": {},
            },
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-structure-metrics", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"bundle_size_histogram\"" in captured.out


def test_dataflow_audit_emits_decision_snapshot(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "decision_snapshot": {
                "format_version": 1,
                "root": ".",
                "decision_surfaces": [],
                "value_decision_surfaces": [],
                "summary": {},
            },
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=["sample.py", "--emit-decision-snapshot", "-"],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"decision_surfaces\"" in captured.out


def test_dataflow_audit_emits_fingerprint_outputs(capsys) -> None:
    class DummyCtx:
        args: list[str] = []

    def runner(*_args, **_kwargs):
        # dataflow-bundle: _args, _kwargs
        return {
            "exit_code": 0,
            "fingerprint_synth_registry": {"version": "synth@1", "entries": []},
            "fingerprint_provenance": [{"path": "x.py", "bundle": ["a"]}],
            "fingerprint_deadness": [{"path": "x.py", "bundle": ["a"], "result": "UNREACHABLE"}],
            "fingerprint_coherence": [
                {
                    "site": {"path": "x.py", "function": "f", "bundle": ["a"]},
                    "result": "UNKNOWN",
                }
            ],
        }

    request = cli.DataflowAuditRequest(
        ctx=DummyCtx(),
        args=[
            "sample.py",
            "--fingerprint-synth-json",
            "-",
            "--fingerprint-provenance-json",
            "-",
            "--fingerprint-deadness-json",
            "-",
            "--fingerprint-coherence-json",
            "-",
        ],
        runner=runner,
    )
    with pytest.raises(typer.Exit) as exc:
        cli._dataflow_audit(request)
    assert exc.value.exit_code == 0
    captured = capsys.readouterr()
    assert "\"bundle\"" in captured.out
    assert "\"version\"" in captured.out
    assert "\"UNREACHABLE\"" in captured.out
    assert "\"fingerprint_coherence\"" not in captured.out
    assert "\"UNKNOWN\"" in captured.out


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


def test_emit_synth_outputs_lists_optional_paths(tmp_path: Path, capsys) -> None:
    root = tmp_path / "out"
    root.mkdir()
    paths_out = {
        "output_root": root,
        "report": root / "dataflow_report.md",
        "dot": root / "graph.dot",
        "plan": root / "plan.json",
        "protocol": root / "protocols.py",
        "refactor": root / "refactor.json",
        "fingerprint_synth": root / "fingerprint_synth.json",
        "fingerprint_provenance": root / "fingerprint_provenance.json",
    }
    paths_out["fingerprint_synth"].write_text("{}")
    paths_out["fingerprint_provenance"].write_text("{}")
    cli._emit_synth_outputs(
        paths_out=paths_out,
        timestamp=None,
        refactor_plan=False,
    )
    output = capsys.readouterr().out
    assert "fingerprint_synth.json" in output
    assert "fingerprint_provenance.json" in output


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


def test_run_structure_diff_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"added_bundles": []}

    baseline = tmp_path / "base.json"
    current = tmp_path / "current.json"
    result = cli.run_structure_diff(
        baseline=baseline,
        current=current,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.STRUCTURE_DIFF_COMMAND
    assert captured["payload"] == {"baseline": str(baseline), "current": str(current)}
    assert captured["root"] == tmp_path
    assert result == {"added_bundles": []}


def test_run_decision_diff_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    baseline = tmp_path / "base.json"
    current = tmp_path / "current.json"
    result = cli.run_decision_diff(
        baseline=baseline,
        current=current,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.DECISION_DIFF_COMMAND
    assert captured["payload"] == {"baseline": str(baseline), "current": str(current)}
    assert result == {"exit_code": 0}


def test_run_structure_reuse_uses_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    snapshot = tmp_path / "snapshot.json"
    lemma = tmp_path / "lemmas.py"
    result = cli.run_structure_reuse(
        snapshot=snapshot,
        min_count=3,
        lemma_stubs=lemma,
        root=tmp_path,
        runner=runner,
    )
    assert captured["command"] == cli.STRUCTURE_REUSE_COMMAND
    assert captured["payload"]["snapshot"] == str(snapshot)
    assert captured["payload"]["min_count"] == 3
    assert captured["payload"]["lemma_stubs"] == str(lemma)
    assert result == {"exit_code": 0}


def test_cli_diff_and_reuse_commands_use_default_runner(capsys) -> None:
    calls: list[str] = []

    def runner(request, root=None):
        calls.append(request.command)
        if request.command == cli.STRUCTURE_REUSE_COMMAND:
            return {"exit_code": 0, "reuse": {"format_version": 1}}
        return {"exit_code": 0, "diff": {"format_version": 1}}

    saved = cli.DEFAULT_RUNNER
    cli.DEFAULT_RUNNER = runner
    try:
        cli.structure_diff(
            baseline=Path("baseline.json"),
            current=Path("current.json"),
            root=None,
        )
        cli.decision_diff(
            baseline=Path("baseline.json"),
            current=Path("current.json"),
            root=None,
        )
        cli.structure_reuse(
            snapshot=Path("snapshot.json"),
            min_count=2,
            lemma_stubs=None,
            root=None,
        )
    finally:
        cli.DEFAULT_RUNNER = saved
    captured = capsys.readouterr().out
    assert "format_version" in captured
    assert cli.STRUCTURE_DIFF_COMMAND in calls
    assert cli.DECISION_DIFF_COMMAND in calls
    assert cli.STRUCTURE_REUSE_COMMAND in calls


def test_emit_structure_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_structure_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out
    assert captured.err == ""


def test_emit_structure_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad snapshot"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "\"exit_code\": 2" in captured.out
    assert "bad snapshot" in captured.err


def test_emit_decision_diff_success(capsys) -> None:
    result = {"exit_code": 0, "diff": {"summary": {"added": 0}}}
    cli._emit_decision_diff(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


def test_emit_decision_diff_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad decision"], "diff": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_decision_diff(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad decision" in captured.err


def test_emit_structure_reuse_success(capsys) -> None:
    result = {"exit_code": 0, "reuse": {"summary": {}}}
    cli._emit_structure_reuse(result)
    captured = capsys.readouterr()
    assert "\"exit_code\": 0" in captured.out


def test_emit_structure_reuse_errors_exit(capsys) -> None:
    result = {"exit_code": 2, "errors": ["bad reuse"], "reuse": {}}
    with pytest.raises(typer.Exit) as exc:
        cli._emit_structure_reuse(result)
    assert exc.value.exit_code == 2
    captured = capsys.readouterr()
    assert "bad reuse" in captured.err
