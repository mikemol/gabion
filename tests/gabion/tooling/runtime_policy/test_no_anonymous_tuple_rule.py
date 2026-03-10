from __future__ import annotations

from pathlib import Path

from gabion.tooling.policy_rules import no_anonymous_tuple_rule as rule
from gabion.tooling.runtime.policy_scan_batch import build_policy_scan_batch_from_sources


# gabion:behavior primary=desired
def test_collect_violations_detects_anonymous_tuple_shapes() -> None:
    source = "\n".join(
        [
            "def emit_pair(value):",
            "    pair = (value, value + 1)",
            "    return pair",
            "",
            "def call_pair(value):",
            "    return consume((value, value + 1))",
            "",
            "def tuple_ctor(values):",
            "    return tuple(values)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    kinds = {item.kind for item in violations}
    assert "tuple_assignment" in kinds
    assert "tuple_argument" in kinds
    assert "tuple_constructor" in kinds
    assert violations[0].fiber_trace


# gabion:behavior primary=desired
def test_collect_violations_accepts_dto_only_shapes() -> None:
    source = "\n".join(
        [
            "from dataclasses import dataclass",
            "",
            "@dataclass(frozen=True)",
            "class PairDTO:",
            "    left: int",
            "    right: int",
            "",
            "def emit_pair(value):",
            "    return PairDTO(left=value, right=value + 1)",
            "",
        ]
    )
    batch = build_policy_scan_batch_from_sources(
        root=Path("."),
        source_by_rel_path={"src/gabion/sample.py": source},
    )

    violations = rule.collect_violations(batch=batch)

    assert violations == []
