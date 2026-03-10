from __future__ import annotations

import json
from pathlib import Path

from gabion.tooling.policy_rules import fiber_filter_processor_contract_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# gabion:behavior primary=desired
def test_collect_violations_detects_loop_branches_and_comprehension_branches() -> None:
    source = "\n".join(
        [
            "def explicit(values):",
            "    for value in values:",
            "        if value:",
            "            yield value",
            "",
            "def comp_filter(values):",
            "    return [value for value in values if value]",
            "",
            "def comp_ifexp(values):",
            "    return [value if value > 0 else 0 for value in values]",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )
    violations = rule.collect_violations(batch=batch)
    kinds = {item.kind for item in violations}
    assert "branch_in_loop_processor" in kinds
    assert "comprehension_filter_branch" in kinds
    assert "comprehension_ifexp_branch" in kinds
    assert violations[0].fiber_trace
    assert violations[0].applicability_bounds is not None
    assert violations[0].counterfactual_boundary is not None


# gabion:behavior primary=desired
def test_collect_violations_accepts_separate_filter_and_processor() -> None:
    source = "\n".join(
        [
            "def filtered(values):",
            "    return (value for value in values)",
            "",
            "def process(values):",
            "    for value in values:",
            "        yield value",
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
                "def sample(values):",
                "    for value in values:",
                "        if value:",
                "            yield value",
                "",
            ]
        ),
    )
    out = root / "artifacts/out/fiber_filter_processor_contract.json"
    exit_code = rule.run(root=root, out=out)
    assert exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["violation_count"] >= 1
    first = payload["violations"][0]
    assert first["fiber_trace"]
    assert first["counterfactual_boundary"]["status"] == "present"
