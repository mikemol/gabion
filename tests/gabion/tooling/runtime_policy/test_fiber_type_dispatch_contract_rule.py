from __future__ import annotations

import ast
import json
from pathlib import Path

from gabion.tooling.policy_rules import fiber_type_dispatch_contract_rule as rule


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_collect_violations_detects_manual_type_routing_forms() -> None:
    source = "\n".join(
        [
            "HANDLERS = {}",
            "",
            "def sample(value: object) -> object:",
            "    if isinstance(value, str):",
            "        return value",
            "    if type(value) is int:",
            "        return value",
            "    match type(value):",
            "        case int:",
            "            return value",
            "        case _:",
            "            return value",
            "    return HANDLERS.get(type(value), lambda _: value)(value)",
            "",
        ]
    )
    tree = ast.parse(source)
    violations = rule.collect_violations(
        rel_path="src/gabion/sample.py",
        source=source,
        tree=tree,
    )
    kinds = {item.kind for item in violations}
    assert "manual_type_guard" in kinds
    assert "match_type_guard" in kinds
    assert "dispatch_table_lookup" in kinds
    assert violations[0].fiber_trace
    assert violations[0].applicability_bounds is not None
    assert violations[0].counterfactual_boundary is not None


def test_collect_violations_accepts_concrete_singledispatch_with_never_base() -> None:
    source = "\n".join(
        [
            "from functools import singledispatch",
            "from gabion.invariants import never",
            "",
            "@singledispatch",
            "def normalize(value: object) -> str:",
            "    never(\"unregistered runtime type\", value_type=type(value).__name__)",
            "",
            "@normalize.register",
            "def _(value: str) -> str:",
            "    return value",
            "",
        ]
    )
    tree = ast.parse(source)
    violations = rule.collect_violations(
        rel_path="src/gabion/sample.py",
        source=source,
        tree=tree,
    )
    assert violations == []


def test_collect_violations_flags_missing_never_base_and_abstract_registration() -> None:
    source = "\n".join(
        [
            "from functools import singledispatch",
            "from collections.abc import Mapping",
            "",
            "@singledispatch",
            "def normalize(value: object) -> object:",
            "    return value",
            "",
            "@normalize.register(Mapping)",
            "def _(value: Mapping[str, object]) -> object:",
            "    return value",
            "",
        ]
    )
    tree = ast.parse(source)
    violations = rule.collect_violations(
        rel_path="src/gabion/sample.py",
        source=source,
        tree=tree,
    )
    kinds = {item.kind for item in violations}
    assert "missing_never_base" in kinds
    assert "abstract_register_type" in kinds


def test_run_writes_fiber_payload(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/sample.py",
        "\n".join(
            [
                "def sample(value: object) -> object:",
                "    if isinstance(value, str):",
                "        return value",
                "    return value",
                "",
            ]
        ),
    )
    out = root / "artifacts/out/fiber_type_dispatch_contract.json"
    exit_code = rule.run(root=root, out=out)
    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["violation_count"] >= 1
    first = payload["violations"][0]
    assert first["fiber_trace"]
    assert first["counterfactual_boundary"]["status"] == "present"
