from __future__ import annotations

from pathlib import Path

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
        ]
    )
    payload = cli.build_dataflow_payload(opts)
    assert payload["paths"] == ["."]
    assert payload["strictness"] == "low"
    assert payload["exclude"] == ["a", "b"]
    assert payload["ignore_params"] == ["x", "y"]
    assert payload["no_recursive"] is True
    assert payload["fail_on_violations"] is True


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
