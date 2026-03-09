from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_rules import fiber_return_shape_contract_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_collect_violations_detects_all_return_shape_kinds() -> None:
    source = "\n".join(
        [
            "def eager(values):",
            "    return [value for value in values]",
            "",
            "def singleton_iter():",
            "    return iter((1,))",
            "",
            "def singleton_generator():",
            "    yield 1",
            "",
            "def _passthrough(value):",
            "    return normalize(value)",
            "",
            "def caller(value):",
            "    return _passthrough(value)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )
    violations = rule.collect_violations(batch=batch)
    kinds = {item.kind for item in violations}
    assert "container_return_prefer_iterator" in kinds
    assert "iterator_return_prefer_item" in kinds
    assert "single_item_return_prefer_inline" in kinds
    assert violations[0].fiber_trace
    assert violations[0].applicability_bounds is not None
    assert violations[0].counterfactual_boundary is not None


# gabion:behavior primary=desired
def test_collect_violations_accepts_non_singleton_and_multi_use_helpers() -> None:
    source = "\n".join(
        [
            "def stream(values):",
            "    for value in values:",
            "        yield value",
            "",
            "def _passthrough(value):",
            "    return normalize(value)",
            "",
            "def caller_one(value):",
            "    return _passthrough(value)",
            "",
            "def caller_two(value):",
            "    return _passthrough(value)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )
    violations = rule.collect_violations(batch=batch)
    assert violations == []


# gabion:behavior primary=desired
def test_run_writes_fiber_payload(tmp_path: Path) -> None:
    root = tmp_path
    _write(
        root / "src/gabion/sample.py",
        "\n".join(
            [
                "def eager(values):",
                "    return list(values)",
                "",
            ]
        ),
    )
    out = root / "artifacts/out/fiber_return_shape_contract.json"
    exit_code = rule.run(root=root, out=out)
    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["violation_count"] >= 1
    first = payload["violations"][0]
    assert first["fiber_trace"]
    assert first["counterfactual_boundary"]["status"] == "present"
