from __future__ import annotations

import ast
import json
import runpy
import sys
from pathlib import Path

import pytest

from gabion.tooling.governance import ambiguity_contract_policy_check as policy
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_collect_violations_respects_boundaries::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check.collect_violations
# gabion:behavior primary=desired
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
        "    match value:\n"
        "        case _:\n"
        "            pass\n"
        "    return value\n\n"
        "def workaround_match(value):\n"
        "    placeholder = ''\n"
        "    matched = False\n"
        "    match value:\n"
        "        case str() as text:\n"
        "            placeholder = text\n"
        "            matched = True\n"
        "    if not matched:\n"
        "        return None\n"
        "    return placeholder\n\n"
        "def workaround_if(value):\n"
        "    normalized = ''\n"
        "    matched = False\n"
        "    if isinstance(value, str):\n"
        "        normalized = value\n"
        "        matched = True\n"
        "    if not matched:\n"
        "        return normalized\n"
        "    return normalized\n\n"
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

    batch = build_policy_scan_batch(root=tmp_path, target_globs=policy.TARGETS)
    violations = policy.collect_violations(batch=batch)
    assert violations
    rule_ids = {item.rule_id for item in violations}
    assert {"ACP-002", "ACP-003", "ACP-004", "ACP-005", "ACP-006", "ACP-007"}.issubset(rule_ids)
    probe_recovery_violations = [
        item for item in violations if item.rule_id == "ACP-006"
    ]
    nullable_contract_violations = [
        item for item in violations if item.rule_id == "ACP-007"
    ]
    assert len(probe_recovery_violations) >= 2
    assert nullable_contract_violations
    assert all("boundary.py" not in item.path for item in violations)
    assert all("__pycache__" not in item.path for item in violations)
    rendered = violations[0].render()
    assert "[ACP-" in rendered
    assert violations[0].key.count(":") >= 2
    rendered_probe = probe_recovery_violations[0].render()
    assert "why:" in rendered_probe
    assert "prefer:" in rendered_probe
    assert "avoid:" in rendered_probe
    rendered_nullable = nullable_contract_violations[0].render()
    assert "nullable contract leaked past ingress" in rendered_nullable
    assert "do not replace None with a custom sentinel" in rendered_nullable


def test_ambiguity_contract_ignores_boundary_dispatch_and_reducer_patterns(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "src" / "gabion" / "analysis" / "good.py",
        "from functools import singledispatch\n\n"
        "@singledispatch\n"
        "def normalize(value):\n"
        "    return None\n\n"
        "@normalize.register(str)\n"
        "def _(value):\n"
        "    return value\n\n"
        "def good(items):\n"
        "    accepted = tuple(\n"
        "        name\n"
        "        for name in (normalize(item) for item in items)\n"
        "        if name is not None\n"
        "    )\n"
        "    return accepted\n",
    )

    batch = build_policy_scan_batch(root=tmp_path, target_globs=policy.TARGETS)
    violations = policy.collect_violations(batch=batch)
    assert not any(item.rule_id == "ACP-006" for item in violations)


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_helper_predicates_cover_all_sentinels::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._module_boundary::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._has_marker::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._annotation_is_dynamic::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._looks_like_guard::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._single_sentinel_stmt
# gabion:behavior primary=desired
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
    assert policy._contains_none_guard(none_guard)
    plain_guard = ast.parse("if value > 0:\n    pass\n").body[0].test
    assert not policy._looks_like_guard(plain_guard)
    assert not policy._contains_none_guard(plain_guard)

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
    wildcard_case = ast.parse("match x:\n    case _:\n        pass\n").body[0].cases[0]
    assert policy._is_fallthrough_case(wildcard_case)
    assert not policy._body_calls_never(wildcard_case.body)
    never_case = ast.parse("match x:\n    case _:\n        never('x')\n").body[0].cases[0]
    assert policy._body_calls_never(never_case.body)


# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_baseline_io_and_run_paths::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._load_baseline::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check._write_baseline::ambiguity_contract_policy_check.py::gabion.tooling.ambiguity_contract_policy_check.run
# gabion:behavior primary=desired
def test_ambiguity_contract_baseline_io_and_run_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "src" / "gabion" / "analysis" / "bad.py",
        "import json\n"
        "from typing import Optional\n\n"
        "def bad(value: Optional[int]) -> Optional[int]:\n"
        "    if value is None:\n"
        "        return None\n"
        "    return json.loads(value)\n",
    )
    baseline = root / policy.DEFAULT_BASELINE_RELATIVE_PATH
    artifact_path = root / policy.ARTIFACT_RELATIVE_PATH

    assert policy.run(root=root, baseline=None, baseline_write=False) == 1
    assert policy.run(root=root, baseline=None, baseline_write=True) == 0
    assert baseline.exists()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "ast" in payload
    assert "grade" in payload
    assert "witness_rows" in payload["grade"]
    assert policy.run(root=root, baseline=None, baseline_write=False) == 0
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
# gabion:behavior primary=desired
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
                "gabion.tooling.governance.ambiguity_contract_policy_check",
                run_name="__main__",
            )
    finally:
        sys.argv = previous_argv
    assert int(exc.value.code or 0) == 0

# gabion:evidence E:call_footprint::tests/test_ambiguity_contract_policy_check.py::test_ambiguity_contract_gate_blocks_on_grade_monotonicity_violation::ambiguity_contract_policy_check.py::gabion.tooling.governance.ambiguity_contract_policy_check.run::grade_monotonicity_semantic.py::gabion.tooling.policy_substrate.grade_monotonicity_semantic.collect_grade_monotonicity
# gabion:behavior primary=desired
def test_ambiguity_contract_gate_blocks_on_grade_monotonicity_violation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "src" / "gabion" / "analysis" / "grade.py",
        "import json\n\n"
        "def strict(value: int) -> object:\n"
        "    return json.loads(str(value))\n",
    )

    assert policy.run(root=root, baseline=None, baseline_write=False) == 1
    artifact = json.loads((root / policy.ARTIFACT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert artifact["grade"]["violation_count"] >= 1
    assert artifact["decisions"]["grade_monotonicity"]["rule_id"] == "grade_monotonicity.new_violations"
    assert artifact["decisions"]["grade_monotonicity"]["outcome"] == "block"
