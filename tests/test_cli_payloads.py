from __future__ import annotations

from pathlib import Path

import pytest
import typer

from gabion import cli


def test_check_builds_payload() -> None:
    payload = cli.build_check_payload(
        paths=None,
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=False,
        exclude=None,
        ignore_params=None,
        transparent_decorators=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=True,
    )
    assert payload["paths"] == ["."]
    assert payload["fail_on_violations"] is True
    assert payload["fail_on_type_ambiguities"] is True
    assert payload["type_audit"] is True


def test_check_payload_strictness_validation() -> None:
    with pytest.raises(typer.BadParameter):
        cli.build_check_payload(
            paths=[Path(".")],
            report=None,
            fail_on_violations=True,
            root=Path("."),
            config=None,
            baseline=None,
            baseline_write=False,
            exclude=None,
            ignore_params=None,
            transparent_decorators=None,
            allow_external=None,
            strictness="medium",
            fail_on_type_ambiguities=False,
        )


def test_check_payload_baseline_write_requires_baseline() -> None:
    payload = cli.build_check_payload(
        paths=[Path(".")],
        report=None,
        fail_on_violations=True,
        root=Path("."),
        config=None,
        baseline=None,
        baseline_write=True,
        exclude=None,
        ignore_params=None,
        transparent_decorators=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=False,
    )
    assert payload["baseline_write"] is None


def test_dataflow_audit_payload_parsing() -> None:
    opts = cli.parse_dataflow_args(
        [
            ".",
            "--strictness",
            "low",
            "--exclude",
            "a,b",
            "--ignore-params",
            "x,y",
            "--no-recursive",
            "--fail-on-violations",
            "--emit-structure-tree",
            "snapshot.json",
            "--emit-structure-metrics",
            "metrics.json",
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["paths"] == ["."]
    assert payload["strictness"] == "low"
    assert payload["exclude"] == ["a", "b"]
    assert payload["ignore_params"] == ["x", "y"]
    assert payload["no_recursive"] is True
    assert payload["fail_on_violations"] is True
    assert payload["structure_tree"] == "snapshot.json"
    assert payload["structure_metrics"] == "metrics.json"


def test_dataflow_payload_baseline_and_transparent() -> None:
    opts = cli.parse_dataflow_args(
        [
            ".",
            "--baseline",
            "baseline.txt",
            "--baseline-write",
            "--transparent-decorators",
            "foo,bar",
            "--fail-on-type-ambiguities",
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["baseline"] == "baseline.txt"
    assert payload["baseline_write"] is True
    assert payload["transparent_decorators"] == ["foo", "bar"]
    assert payload["fail_on_type_ambiguities"] is True


def test_refactor_protocol_payload(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=["a", "b"],
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=["alpha"],
        compatibility_shim=True,
        rationale="use bundle",
    )
    assert payload["protocol_name"] == "Bundle"
    assert payload["bundle"] == ["a", "b"]
    assert payload["target_functions"] == ["alpha"]
    assert payload["compatibility_shim"] is True


def test_refactor_payload_infers_bundle(tmp_path: Path) -> None:
    payload = cli.build_refactor_payload(
        protocol_name="Bundle",
        bundle=None,
        field=["a:int", "b:str"],
        target_path=tmp_path / "sample.py",
        target_functions=[],
        compatibility_shim=False,
        rationale=None,
    )
    assert payload["bundle"] == ["a", "b"]


def test_run_check_uses_runner_dispatch(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def runner(request, *, root=None):
        captured["command"] = request.command
        captured["payload"] = request.arguments[0]
        captured["root"] = root
        return {"exit_code": 0}

    result = cli.run_check(
        paths=[tmp_path],
        report=None,
        fail_on_violations=True,
        root=tmp_path,
        config=None,
        baseline=None,
        baseline_write=False,
        exclude=None,
        ignore_params=None,
        transparent_decorators=None,
        allow_external=None,
        strictness=None,
        fail_on_type_ambiguities=True,
        runner=runner,
    )
    assert result["exit_code"] == 0
    assert captured["command"] == cli.DATAFLOW_COMMAND
    assert captured["payload"]["paths"] == [str(tmp_path)]
    assert captured["root"] == tmp_path
