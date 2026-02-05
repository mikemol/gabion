from __future__ import annotations

from pathlib import Path
import sys


def _load():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from gabion.analysis import dataflow_audit as da

    return da


def test_lint_location_parser_and_bundle_lines() -> None:
    da = _load()
    parsed = da._parse_lint_location("mod.py:10:4-10:6: f -> g forwards a,b")
    assert parsed is not None
    path, lineno, col, remainder = parsed
    assert path == "mod.py"
    assert lineno == 10
    assert col == 4
    assert "f -> g" in remainder
    lines = da._lint_lines_from_bundle_evidence(
        ["mod.py:10:4-10:6: f -> g forwards a,b"]
    )
    assert lines == ["mod.py:10:4: GABION_BUNDLE_UNDOC f -> g forwards a,b"]
    assert da._parse_lint_location("bad line") is None
    assert da._parse_lint_location("mod.py:x:y: nope") is None


def test_lint_lines_from_type_and_exception_evidence() -> None:
    da = _load()
    type_lines = da._lint_lines_from_type_evidence(
        ["mod.py:5:2: f.a -> g.b expects int"]
    )
    assert type_lines == ["mod.py:5:2: GABION_TYPE_FLOW f.a -> g.b expects int"]
    assert da._parse_exception_path_id("bad") is None
    assert da._parse_exception_path_id("mod.py:f:E0:bad:1:raise") is None
    exception_entries = [
        {
            "exception_path_id": "mod.py:f:E0:3:1:raise",
            "exception_name": "NeverRaise",
            "protocol": "never",
            "status": "FORBIDDEN",
        }
    ]
    lint_lines = da._exception_protocol_lint_lines(exception_entries)
    assert lint_lines == [
        "mod.py:3:1: GABION_EXC_NEVER never-throw exception NeverRaise (status=FORBIDDEN)"
    ]


def test_lint_lines_from_constant_and_unused_smells() -> None:
    da = _load()
    constant_smell = (
        "mod.py:f.a only observed constant 1 across 2 non-test call(s) "
        "(e.g. mod.py:10:4:f)"
    )
    const_lines = da._lint_lines_from_constant_smells([constant_smell])
    assert const_lines == [
        "mod.py:10:4: GABION_CONST_FLOW "
        "mod.py:f.a only observed constant 1 across 2 non-test call(s) (e.g. mod.py:10:4:f)"
    ]
    unused_smell = "mod.py:12:3:f passes param x to unused mod.py:g.y"
    unused_lines = da._lint_lines_from_unused_arg_smells([unused_smell])
    assert unused_lines == [
        "mod.py:12:3: GABION_UNUSED_ARG f passes param x to unused mod.py:g.y"
    ]
