from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

from gabion.tooling import ambiguity_contract_policy_check as policy


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_collect_violations_respects_boundaries::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check.collect_violations
def test_ambiguity_contract_collect_violations_respects_boundaries(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "bad.py",
        "from typing import Any, Optional, Union\n\n"
        "def bad(value: Optional[int]) -> Optional[int]:\n"
        "    typed: Union[int, None] = value\n"
        "    _ = len([value])\n"
        "    if value and value > 0:\n"
        "        typed = value\n"
        "    if isinstance(value, int):\n"
        "        return None\n"
        "    if value is None:\n"
        "        typed = 0\n"
        "    return value\n\n"
        "async def bad_async(value: Any) -> int:\n"
        "    if value is None:\n"
        "        return []\n"
        "    return 0\n",
    )
    _write(
        tmp_path / "src" / "gabion" / "synthesis" / "boundary.py",
        "# gabion:ambiguity_boundary_module\n"
        "from typing import Optional\n\n"
        "def allowed(value: Optional[int]) -> Optional[int]:\n"
        "    marker: Optional[int] = value\n"
        "    if isinstance(value, int):\n"
        "        return value\n"
        "    return value\n",
    )
    _write(
        tmp_path / "src" / "gabion" / "refactor" / "__pycache__" / "ignored.py",
        "def ignored() -> None:\n"
        "    pass\n",
    )

    violations = policy.collect_violations(tmp_path)
    assert violations
    rule_ids = {item.rule_id for item in violations}
    assert {"ACP-002", "ACP-003", "ACP-004"}.issubset(rule_ids)
    assert all("boundary.py" not in item.path for item in violations)
    assert all("__pycache__" not in item.path for item in violations)
    rendered = violations[0].render()
    assert "[ACP-" in rendered
    assert violations[0].key.count(":") >= 2


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_helper_predicates_cover_all_sentinels::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._module_boundary::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._has_marker::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._annotation_is_dynamic::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._looks_like_guard::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._single_sentinel_stmt
def test_ambiguity_contract_helper_predicates_cover_all_sentinels() -> None:
    assert policy._module_boundary(
        [
            "\"\"\"doc\"\"\"",
            "# gabion:ambiguity_boundary_module",
        ]
    )
    assert not policy._module_boundary(["\"\"\"doc\"\"\"", "value = 1"])
    assert policy._has_marker(
        ["# gabion:ambiguity_boundary", "", "def f():", "    pass"],
        3,
        policy.FUNCTION_MARKER,
    )
    assert not policy._has_marker(["value = 1", "def f():", "    pass"], 2, policy.FUNCTION_MARKER)
    assert not policy._has_marker(["", ""], 1, policy.FUNCTION_MARKER)
    assert policy._Visitor(rel_path="x.py", source_lines=["value = 1"])._scope_boundary is False
    assert (
        policy._Visitor(
            rel_path="x.py",
            source_lines=["# gabion:ambiguity_boundary_module"],
        )._scope_boundary
        is True
    )

    assert policy._annotation_is_dynamic(ast.parse("x: Any").body[0].annotation)
    assert policy._annotation_is_dynamic(ast.parse("x: Optional[int]").body[0].annotation)
    assert policy._annotation_is_dynamic(ast.parse("x: int | str").body[0].annotation)
    assert not policy._annotation_is_dynamic(ast.parse("x: int").body[0].annotation)
    assert policy._annotation_is_dynamic(ast.parse("x: tuple[Any, int]").body[0].annotation)
    assert not policy._annotation_is_dynamic(ast.parse("x: dict[str, int]").body[0].annotation)

    isinstance_guard = ast.parse("if isinstance(value, int):\n    pass\n").body[0].test
    assert policy._looks_like_guard(isinstance_guard)
    none_guard = ast.parse("if value is None:\n    pass\n").body[0].test
    assert policy._looks_like_guard(none_guard)
    plain_guard = ast.parse("if value > 0:\n    pass\n").body[0].test
    assert not policy._looks_like_guard(plain_guard)

    assert policy._single_sentinel_stmt(
        ast.parse("def f():\n    return\n").body[0].body  # type: ignore[union-attr]
    ) == "return None"
    assert policy._single_sentinel_stmt(
        ast.parse("def f():\n    return None\n").body[0].body  # type: ignore[union-attr]
    ) == "return None"
    assert policy._single_sentinel_stmt(
        ast.parse("def f():\n    return []\n").body[0].body  # type: ignore[union-attr]
    ) == "return []"
    assert policy._single_sentinel_stmt(ast.parse("for x in y:\n    continue\n").body[0].body) == "continue"
    assert policy._single_sentinel_stmt(ast.parse("if True:\n    pass\n").body[0].body) == "pass"
    assert policy._single_sentinel_stmt(
        ast.parse("def f():\n    return 1\n").body[0].body  # type: ignore[union-attr]
    ) is None
    assert policy._single_sentinel_stmt(
        ast.parse("def f():\n    x = 1\n    return 1\n").body[0].body  # type: ignore[union-attr]
    ) is None


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_baseline_io_and_run_paths::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._load_baseline::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._write_baseline::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check.run
def test_ambiguity_contract_baseline_io_and_run_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "src" / "gabion" / "analysis" / "bad.py",
        "from typing import Optional\n"
        "def bad(value: Optional[int]) -> Optional[int]:\n"
        "    if value is None:\n"
        "        return None\n"
        "    return value\n",
    )
    baseline = tmp_path / "baseline.json"

    with pytest.raises(SystemExit):
        policy.run(root=root, baseline=None, baseline_write=True)
    assert policy.run(root=root, baseline=None, baseline_write=False) == 1
    assert policy.run(root=root, baseline=baseline, baseline_write=True) == 0
    assert baseline.exists()
    assert policy.run(root=root, baseline=baseline, baseline_write=False) == 0

    missing = tmp_path / "missing.json"
    assert policy._load_baseline(missing) == set()

    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text("[]\n", encoding="utf-8")
    assert policy._load_baseline(non_dict) == set()

    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        (
            "{\n"
            "  \"violations\": [\n"
            "    {\"rule_id\": \"ACP-001\", \"path\": \"a.py\", \"qualname\": \"f\", \"line\": 3},\n"
            "    \"bad\",\n"
            "    {\"rule_id\": 1, \"path\": \"a.py\", \"qualname\": \"f\", \"line\": 3}\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    assert policy._load_baseline(mixed) == {"ACP-001:a.py:f"}
    non_list = tmp_path / "non_list.json"
    non_list.write_text("{\"violations\": {}}\n", encoding="utf-8")
    assert policy._load_baseline(non_list) == set()


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_main_and_module_entrypoint::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check.main
def test_ambiguity_contract_main_and_module_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "src" / "gabion" / "analysis" / "clean.py",
        "# gabion:ambiguity_boundary_module\n"
        "def clean() -> int:\n"
        "    return 1\n",
    )
    baseline = tmp_path / "baseline.json"

    assert policy.main(["--root", str(root)]) == 0
    assert policy.main(
        ["--root", str(root), "--baseline", str(baseline), "--baseline-write"]
    ) == 0
    assert policy.main(["--root", str(root), "--baseline", str(baseline)]) == 0

    previous_argv = list(sys.argv)
    try:
        sys.argv = [
            "ambiguity_contract_policy_check",
            "--root",
            str(root),
            "--baseline",
            str(baseline),
        ]
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(
                "gabion.tooling.ambiguity_contract_policy_check",
                run_name="__main__",
            )
    finally:
        sys.argv = previous_argv
    assert int(exc.value.code or 0) == 0
